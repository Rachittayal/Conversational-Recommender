from typing import Optional
from pydantic import BaseModel,Field,field_validator


class Message(BaseModel):
    role: str=Field(..., pattern="^(user|assistant)$")
    content: str=Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)
    
    @field_validator("messages")
    @classmethod
    def last_message_must_be_user(cls,msgs):
        if msgs[-1].role != "user":
            raise ValueError("Last message must be from user")

        return msgs


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: Optional[list[Recommendation]] = []
    end_of_conversation: bool = False
    

class HealthResponse(BaseModel):
    status: str = "ok"

class CatalogItem(BaseModel):
    model_config = {"frozen": True}
 
    name: str
    url: str                                            
    test_type: str                                      
    all_types: list[str] = Field(default_factory=list)  
    description: str = ""
    duration: Optional[str] = None
    languages: list[str] = Field(default_factory=list)
    job_levels: list[str] = Field(default_factory=list)
    remote: bool = False
    adaptive: bool = False

    
    def to_recommendation(self) -> Recommendation:
        return Recommendation(
            name=self.name,
            url=self.url,
            test_type=self.test_type,
        )

    def to_embedding_text(self) -> str:
        from config import CODE_TO_LABEL

        label=CODE_TO_LABEL.get(self.test_type, self.test_type)

        parts=[f"Name: {self.name}", f"Type: {label}"]

        if self.description:
            parts.append(f"Description: {self.description}")

        if self.job_levels:
            parts.append(f"Job levels: {', '.join(self.job_levels)}")

        if self.languages:
            parts.append(f"Languages: {', '.join(self.languages[:5])}")

        if self.remote:
            parts.append("Supports remote testing")

        if self.adaptive:
            parts.append("Adaptive format")

        if self.duration:
            parts.append(f"Duration: {self.duration}")

        return " | ".join(parts)
    
class ContextSlots(BaseModel):
    role: Optional[str] = None        
    domain: Optional[str] = None      
    seniority: Optional[str] = None   
    use_case: Optional[str] = None    
    test_types: list[str] = Field(default_factory=list)  
    language: Optional[str] = None 
    remote: Optional[bool] = None    
 
 
class ConversationAnalysis(BaseModel):
    action: str 
 
    context_slots: ContextSlots = Field(default_factory=ContextSlots)
 
    missing_fields: list[str] = Field(default_factory=list)
 
    clarifying_question: str = ""
 
    items_to_compare: list[str] = Field(default_factory=list)
 
    refine_add: list[str] = Field(default_factory=list)
    refine_remove: list[str] = Field(default_factory=list)
    is_final: bool = False
