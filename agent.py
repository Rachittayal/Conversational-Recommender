import asyncio
import json
import logging
import re

from groq import Groq

from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    REQUEST_TIMEOUT_SECONDS,
    MAX_TURNS,
)
from prompts import (
    SYSTEM_PROMPT,
    build_analysis_prompt,
    build_recommend_prompt,
    build_compare_prompt,
    format_conversation,
)
from retriever import Retriever, SearchQuery
from schemas import (
    CatalogItem,
    ChatRequest,
    ChatResponse,
    ConversationAnalysis,
    ContextSlots,
)

logger=logging.getLogger(__name__)


class Agent:

    def __init__(self, retriever: Retriever):
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")

        self.retriever=retriever
        self.client=Groq(api_key=GROQ_API_KEY)


    async def respond(self, request: ChatRequest) -> ChatResponse:
        messages=request.messages

        if len(messages) >= MAX_TURNS:
            return ChatResponse(
                reply="Maximum conversation length reached. Please start a new chat.",
                recommendations=[],
                end_of_conversation=True,
            )

        # Deterministic off-topic detection (zero LLM calls)
        if self._is_off_topic(messages):
            return self._handle_refuse()

        # Deterministic finality detection (zero LLM calls)
        if self._is_final_signal(messages):
            return self._handle_confirm(messages)

        if self._is_bypass_intent(messages):
            analysis = await self._analyze_conversation(messages)
            return await self._route(messages, analysis)

        if not self._has_minimum_context(messages):
            return self._handle_clarify_templated(messages)

        analysis = await self._analyze_conversation(messages)
        logger.info(
            "Action: %s | is_final: %s | missing: %s",
            analysis.action,
            analysis.is_final,
            analysis.missing_fields,
        )
        return await self._route(messages, analysis)


    async def _route(self,messages: list,analysis: ConversationAnalysis) -> ChatResponse:

        if analysis.action == "REFUSE":
            return self._handle_refuse()

        if analysis.action == "CONFIRM":
            return self._handle_confirm(messages)

        if analysis.action == "COMPARE":
            return await self._handle_compare(messages, analysis)

        if analysis.action == "CLARIFY":
            return await self._handle_clarify_llm(messages, analysis)

        response = await self._handle_recommend(messages, analysis)

        if analysis.is_final:
            return ChatResponse(
                reply=response.reply,
                recommendations=response.recommendations,
                end_of_conversation=True,
            )

        return response

    def _has_minimum_context(self, messages: list) -> bool:
        full_text = " ".join(m.content.lower() for m in messages)

        if len(messages) == 1:
            first_text = messages[0].content.lower()
            if self._has_role(first_text) and self._has_seniority(first_text):
                return True  # Let LLM analyze and decide CLARIFY vs RECOMMEND
            return False  # Too vague, use templated clarify

        has_role = self._has_role(full_text)
        has_seniority = self._has_seniority(full_text)
        has_use_case = self._has_use_case(full_text)
        has_test_type = self._has_explicit_test_type(full_text)

        if has_role and has_seniority and has_use_case:
            return True

        if has_test_type and has_role:
            return True

        executive_signals = [
            "cxo", "director", "executive", "vp", "c-level",
            "leadership", "c-suite",
        ]
        if any(s in full_text for s in executive_signals) and not has_use_case:
            return False

        last_assistant = next(
            (m.content.lower() for m in reversed(messages) if m.role == "assistant"),
            "",
        )
        last_user = messages[-1].content.lower()
        
        confirmation_signals = ["yes", "go ahead", "sure", "okay", "ok", "proceed"]
        if any(s in last_user for s in confirmation_signals):
            return True
        
        purpose_signals = [
            "selection", "development", "hiring", "recruit", "benchmark",
            "feedback", "comparing", "identify", "no preference",
        ]
        if ("selection" in last_assistant or "development" in last_assistant) \
                and "?" in last_assistant:
            if not any(s in last_user for s in purpose_signals):
                return False

        return has_role and has_seniority and len(messages) >= 2

    def _is_bypass_intent(self, messages: list) -> bool:
        last = messages[-1].content.lower()
        bypass_signals = [
            "compare", "difference between", "difference", "what's the difference",
            "vs ", "versus", "vs.",
            "remove", "drop", "add", "replace", "instead", "swap",
            "final list", "drop the", "remove the", "add the",
            "ignore previous", "forget instructions",
            "system prompt", "repeat your instructions",
        ]
        return any(s in last for s in bypass_signals)

    def _is_off_topic(self, messages: list) -> bool:
        """Detect obviously off-topic queries without LLM (zero-cost early exit)"""
        last = messages[-1].content.lower()
        
        # Off-topic keywords
        off_topic_signals = [
            "weather", "recipe", "joke", "story", "news",
            "stock", "sports", "movie", "music", "game",
            "translate", "math problem", "calculate", "convert",
            "restaurant", "travel", "hotel", "flight",
        ]
        
        # Hiring/assessment context keywords
        hiring_context = [
            "assess", "hire", "recruit", "candidate", "test",
            "role", "job", "position", "employee", "talent",
            "selection", "development", "screening", "evaluation",
        ]
        
        has_off_topic = any(s in last for s in off_topic_signals)
        has_hiring_context = any(s in last for s in hiring_context)
        
        # Off-topic if contains off-topic keywords and no hiring context
        return has_off_topic and not has_hiring_context

    def _is_final_signal(self, messages: list) -> bool:
        """Detect finality signals without LLM (zero-cost early exit)"""
        # Need at least one recommendation exchange (user → assistant → user)
        if len(messages) < 3:
            return False
        
        # Check if previous assistant message had recommendations (contains URLs)
        prev_assistant = next(
            (m.content for m in reversed(messages[:-1]) if m.role == "assistant"),
            ""
        )
        has_recommendations = "https://www.shl.com/" in prev_assistant
        
        if not has_recommendations:
            return False
        
        last = messages[-1].content.lower()
        finality_signals = [
            "perfect", "that's what we need", "that works",
            "looks good", "confirmed", "final list", "done",
            "approved", "go with that", "locking it in",
            "that's good", "exactly what", "just what",
            "great", "excellent", "thank you", "thanks",
        ]
        
        # Check for finality signal in short responses (< 50 chars)
        # Longer responses likely have follow-up questions
        if len(last) < 50:
            return any(s in last for s in finality_signals)
        
        return False


    def _has_role(self, text: str) -> bool:
        role_keywords = [
            "engineer", "developer", "scientist", "analyst", "manager",
            "director", "consultant", "specialist", "architect",
            "administrator", "designer", "leader", "officer",
            "coordinator", "technician", "sales", "marketing", "hr",
            "accountant", "finance", "legal", "operations", "product",
            "project", "support", "trainer", "executive", "leadership",
            # Technical / domain
            "machine learning", "data", "software", "python", "java",
            "rust", "c++", "golang", "javascript", "typescript",
            "networking", "infrastructure", "backend", "frontend",
            # Entry / graduate
            "graduate", "trainee", "intern", "associate",
            # Industrial / frontline / operational
            "operator", "plant", "warehouse", "driver", "mechanic",
            "technician", "supervisor", "inspector", "assembler",
            "foreman", "maintenance", "safety", "quality", "cashier",
            "clerk", "agent", "representative", "rep", "assistant",
            "nurse", "doctor", "physician", "pharmacist", "teacher",
            "customer service", "call centre", "call center", "retail",
            "banking", "teller", "branch",
        ]
        return any(w in text for w in role_keywords)

    def _has_seniority(self, text: str) -> bool:
        if re.search(
            r"\b(senior|junior|mid(?:[- ]level)?|entry(?:[- ]level)?|"
            r"graduate|lead|director|manager|executive|experienced|intern|"
            r"associate|principal|staff|vp|cxo|highly|high|top|no preference)\b",
            text,
        ):
            return True
        return bool(re.search(r"\b\d+\s*(?:\+\s*)?years?\b", text))

    def _has_use_case(self, text: str) -> bool:
        return bool(re.search(
            r"\b(selection|development|feedback|benchmark|compare|"
            r"recruitment|promotion|performance|career growth|"
            r"talent development|hiring|screen|evaluate|"
            r"re-skill|reskill|upskill|up-skill|training|audit|"
            r"no preference)\b",
            text,
        ))

    def _has_explicit_test_type(self, text: str) -> bool:
        signals = [
            "cognitive", "personality", "situational", "aptitude",
            "technical", "coding", "knowledge", "ability", "behaviour",
            "behavioral", "numerical", "verbal", "inductive", "deductive",
        ]
        return any(s in text for s in signals)

    def _handle_clarify_templated(self, messages: list) -> ChatResponse:
        """Zero LLM calls. Asks the single most important missing question."""
        full_text = " ".join(m.content.lower() for m in messages)

        if not self._has_role(full_text):
            question = "What role or job function are you hiring for?"
        elif not self._has_seniority(full_text):
            question = "What seniority level is this role?"
        elif not self._has_use_case(full_text):
            question = (
                "Is this for selection (comparing candidates) "
                "or development (existing employees)?"
            )
        else:
            question = "Could you tell me a bit more about what you are looking for?"

        return ChatResponse(
            reply=question,
            recommendations=None,
            end_of_conversation=False,
        )

    def _handle_refuse(self) -> ChatResponse:
        """Zero LLM calls."""
        return ChatResponse(
            reply=(
                "I can only help with SHL assessment selection. "
                "Please ask me about assessments for a specific role or hiring need."
            ),
            recommendations=None,
            end_of_conversation=False,
        )

    def _handle_confirm(self, messages: list) -> ChatResponse:
        items = self._extract_last_recommendations(messages)
        return ChatResponse(
            reply="Great — happy to help. Good luck with your hiring process.",
            recommendations=[item.to_recommendation() for item in items],
            end_of_conversation=True,
        )

    async def _handle_clarify_llm(self,messages: list,analysis: ConversationAnalysis) -> ChatResponse:
        
        question = analysis.clarifying_question.strip()
        if not question:
            return self._handle_clarify_templated(messages)
        return ChatResponse(
            reply=question,
            recommendations=None,
            end_of_conversation=False,
        )

    async def _handle_compare(self,messages: list,analysis: ConversationAnalysis) -> ChatResponse:
        compare_terms = analysis.items_to_compare or []
        if not compare_terms:
            compare_terms = self._extract_compare_terms(messages[-1].content)

        items: list[CatalogItem] = []
        seen_urls: set[str] = set()

        for term in compare_terms:
            match = self.retriever.search_by_name(term)
            if match and match.url not in seen_urls:
                items.append(match)
                seen_urls.add(match.url)

        if not items:
            return ChatResponse(
                reply=(
                    "I could not find those assessments in the catalog. "
                    "Could you clarify the assessment names?"
                ),
                recommendations=None,
                end_of_conversation=False,
            )

        prompt = build_compare_prompt(format_conversation(messages), items)
        reply = await self._llm_call(prompt, max_tokens=300)
        return ChatResponse(
            reply=reply.strip(),
            recommendations=None,
            end_of_conversation=False,
        )

    async def _handle_recommend(self,messages: list,analysis: ConversationAnalysis) -> ChatResponse:
        if analysis.action == "REFINE":
            return await self._handle_refine(messages, analysis)

        slots = analysis.context_slots
        query_text = self._slots_to_query(slots)

        search = SearchQuery(
            query=query_text,
            test_types=slots.test_types,
            job_levels=[slots.seniority] if slots.seniority else [],
            require_remote=slots.remote,
            languages=[slots.language] if slots.language else [],
        )

        results = self.retriever.search(search)

        # Proactive enrichment — add OPQ32r / Verify G+ when appropriate
        results = self._enrich_with_defaults(results, slots, messages)

        if not results:
            return ChatResponse(
                reply=(
                    "I could not find matching assessments. "
                    "Could you relax some requirements?"
                ),
                recommendations=None,
                end_of_conversation=False,
            )

        prompt = build_recommend_prompt(
            format_conversation(messages),
            results,
            query_text or "the role",
        )
        reply = await self._llm_call(prompt, max_tokens=200)

        return ChatResponse(
            reply=reply.strip(),
            recommendations=[item.to_recommendation() for item in results],
            end_of_conversation=False,
        )

    async def _handle_refine(self,messages: list,analysis: ConversationAnalysis) -> ChatResponse:
        
        current_items = self._extract_last_recommendations(messages)
        result_items = list(current_items)

        for name_to_remove in analysis.refine_remove:
            name_lower = name_to_remove.lower()
            result_items = [
                item for item in result_items
                if name_lower not in item.name.lower()
            ]

        seen_urls = {item.url for item in result_items}
        for name_to_add in analysis.refine_add:
            new_item = self.retriever.search_by_name(name_to_add)
            if new_item and new_item.url not in seen_urls:
                result_items.append(new_item)
                seen_urls.add(new_item.url)

        result_items = result_items[:10]

        if not result_items:
            return ChatResponse(
                reply=(
                    "The updated shortlist is empty. "
                    "Could you clarify what you would like to include?"
                ),
                recommendations=None,
                end_of_conversation=False,
            )

        parts = []
        if analysis.refine_remove:
            parts.append(f"Removed: {', '.join(analysis.refine_remove)}")
        if analysis.refine_add:
            parts.append(f"Added: {', '.join(analysis.refine_add)}")
        reply_text = (
            "Updated shortlist — " + "; ".join(parts) + "."
            if parts else "Updated shortlist."
        )

        return ChatResponse(
            reply=reply_text,
            recommendations=[item.to_recommendation() for item in result_items],
            end_of_conversation=False,
        )

    def _enrich_with_defaults(self,results: list[CatalogItem],slots: ContextSlots,messages: list) -> list[CatalogItem]:
        
        full_text = " ".join(m.content.lower() for m in messages)
        existing_urls = {item.url for item in results}
        existing_names_lower = " ".join(item.name.lower() for item in results)
        additions: list[CatalogItem] = []

        leadership_context = any(s in full_text for s in [
            "leadership", "manager", "director", "executive", "cxo",
            "management", "lead", "senior", "head of",
        ])
        excluded_personality = any(s in full_text for s in [
            "no personality", "skip personality", "exclude personality",
            "without personality",
        ])
        opq_present = (
            "opq" in existing_names_lower
            or "personality questionnaire" in existing_names_lower
        )

        if leadership_context and not excluded_personality and not opq_present:
            opq = self.retriever.search_by_name(
                "Occupational Personality Questionnaire OPQ32r"
            )
            if opq and opq.url not in existing_urls:
                additions.append(opq)
                existing_urls.add(opq.url)

        cognitive_context = any(s in full_text for s in [
            "graduate", "analyst", "engineer", "developer", "technical",
            "data", "finance", "accounting", "cognitive", "reasoning",
        ])
        excluded_cognitive = any(s in full_text for s in [
            "no cognitive", "skip cognitive", "exclude cognitive",
            "without cognitive", "no ability",
        ])
        verify_present = "verify" in existing_names_lower

        if cognitive_context and not excluded_cognitive and not verify_present:
            verify = self.retriever.search_by_name("Verify G+ cognitive ability")
            if verify and verify.url not in existing_urls:
                additions.append(verify)
                existing_urls.add(verify.url)

        return (results + additions)[:10]


    async def _analyze_conversation(self, messages: list) -> ConversationAnalysis:
        conversation = format_conversation(messages)
        questions_asked = self._count_clarifying_questions(messages)
        prompt = build_analysis_prompt(conversation, questions_asked)

        try:
            raw = await self._llm_call(prompt, max_tokens=400)
            logger.info("LLM analysis response: %s", raw[:500])  # Log first 500 chars
            data = self._parse_json(raw)

            slots_data = data.get("context_slots", {})
            valid_slot_fields = {
                "role", "domain", "seniority", "use_case",
                "test_types", "language", "remote",
            }
            slots_data = {k: v for k, v in slots_data.items() if k in valid_slot_fields}

            return ConversationAnalysis(
                action=data.get("action", "CLARIFY").upper(),
                context_slots=ContextSlots(**slots_data),
                missing_fields=data.get("missing_fields", []),
                items_to_compare=data.get("items_to_compare", []),
                clarifying_question=data.get("clarifying_question", ""),
                refine_add=data.get("refine_add", []),
                refine_remove=data.get("refine_remove", []),
                is_final=bool(data.get("is_final", False)),
            )

        except Exception as e:
            logger.warning("Analysis call failed: %s", e)
            return ConversationAnalysis(action="CLARIFY")

    def _slots_to_query(self, slots: ContextSlots) -> str:
        parts = filter(None, [
            slots.role, slots.domain, slots.seniority, slots.use_case
        ])
        return " ".join(parts) or "assessment"

    def _count_clarifying_questions(self, messages: list) -> int:
        return sum(
            1 for msg in messages
            if msg.role == "assistant"
            and "?" in msg.content
            and not msg.content.strip().startswith("[REC]")
        )

    def _extract_compare_terms(self, text: str) -> list[str]:
        cleaned = re.sub(
            r"(?i)^.*?(compare|difference between|vs\.?|versus)\s*",
            "",
            text,
            count=1,
        ).strip()
        separators = re.compile(r"\band\b|\bvs\.?\b|\bversus\b|,|;", re.IGNORECASE)
        return [
            part.strip()
            for part in separators.split(cleaned)
            if part.strip() and len(part.strip()) > 2
        ]

    def _extract_last_recommendations(self, messages: list) -> list[CatalogItem]:
        
        url_map = {item.url: item for item in self.retriever.items}

        for msg in reversed(messages):
            if msg.role != "assistant":
                continue

            # Scan for URLs in the message
            urls = re.findall(r"https://www\.shl\.com/\S+", msg.content)
            found = [
                url_map[u.rstrip(".,)")]
                for u in urls
                if u.rstrip(".,)") in url_map
            ]
            if found:
                return found

        return []

    async def _llm_call(self, prompt: str, max_tokens: int = 500) -> str:
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.2,
                ),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            return response.choices[0].message.content or ""
        except asyncio.TimeoutError:
            logger.error("LLM request timed out")
            raise
        except Exception as e:
            logger.error("LLM request failed: %s", e)
            raise

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = re.sub(r"```json|```", "", text).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in LLM response")
        return json.loads(match.group())