
# Nodes with PydanticOutputParsers for HF

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from state import RelevanceState
from llm import llm

class FeatureExtraction(BaseModel):
    features: list[str]

# feature_llm = llm.with_structured_output(FeatureExtraction)
feature_parser = PydanticOutputParser(
    pydantic_object=FeatureExtraction
)

# Extract_product_features
def extract_product_features(state: RelevanceState):

    prompt = f"""
You are an experienced patent analyst.

Extract only the technical features from the product description.

Rules
- Extract only technical features.
- One feature per item.
- Keep each feature concise.
- Preserve technical terminology.

{feature_parser.get_format_instructions()}

Product Description

{state["product_description"]}
"""
    response = llm.invoke(
        [
            HumanMessage(content=prompt)
        ]
    )
    print("=" * 80)
    print(response.content)
    print("=" * 80)
    result = feature_parser.parse(response.content)
    state["product_features"] = result.features
    return state


class FeatureExtraction(BaseModel):
    features: list[str]

# structured_feature_llm = llm.with_structured_output(FeatureExtraction)

feature_parser = PydanticOutputParser(
    pydantic_object=FeatureExtraction
)

# Extract patent features
def extract_patent_features(state: RelevanceState):

    prompt = f"""
You are an experienced patent analyst.

Your task is to extract the important technical features from the
following patent claims.

Instructions:

- Use the ALL CLAIMS and the Independent Claim as the primary source.
- Title and Abstract are only for context.
- Extract only technical features.
- Ignore legal language.
- Do not infer features that are not explicitly disclosed.
- Keep each feature concise.
- One feature per list item.

Return your answer in the following format:

{feature_parser.get_format_instructions()}

Patent Information

Publication Number:
{state["publication_number"]}

Title:
{state["title"]}

Abstract:
{state["abstract"]}

Independent Claim (Reference Only):
{state["independent_claim"]}

All Claims (Primary Source):
{state["all_claims"]}
"""

    response = llm.invoke(
        [
            HumanMessage(content=prompt)
        ]
    )

    try:
        result = feature_parser.parse(response.content)
        state["patent_features"] = result.features

    except Exception as e:
        print("Patent Feature Parsing Error:", e)
        print(response.content)
        state["patent_features"] = []

    return state


# Compare Features

class FeatureMatch(BaseModel):
    product_feature: str
    patent_feature: str
    match_type: Literal[
        "Exact Match",
        "Partial Match",
        "No Match"
    ]
    reasoning: str


class ComparisonOutput(BaseModel):
    comparisons: list[FeatureMatch]

comparison_parser = PydanticOutputParser(
    pydantic_object=ComparisonOutput
)    

# comparison_llm = llm.with_structured_output(ComparisonOutput)

def compare_features(state: RelevanceState):

    prompt = f"""
You are an experienced patent analyst.

Your task is to compare the extracted product features with the
technical features extracted from the patent.

Instructions:

1. Compare EVERY product feature individually.

2. For each product feature:
   - Find the closest matching patent feature.
   - If no suitable feature exists, use "None".

3. Classify the match as ONLY one of:
   - Exact Match
   - Partial Match
   - No Match

Definitions:

Exact Match
- The patent explicitly discloses the same technical feature.

Partial Match
- The patent discloses a similar feature,
  equivalent functionality,
  or a closely related implementation.

No Match
- The feature is not disclosed in the patent.

Additional Instructions:

- Compare only technical features.
- Keep the reasoning to one concise sentence.

Return your answer in the following format:

{comparison_parser.get_format_instructions()}

Product Features:

{state["product_features"]}

Patent Features:

{state["patent_features"]}
"""

    response = llm.invoke(
        [
            HumanMessage(content=prompt)
        ]
    )

    try:
        result = comparison_parser.parse(response.content)
        state["comparison"] = result.comparisons

    except Exception as e:
        print("Comparison Parsing Error:", e)
        print(response.content)
        state["comparison"] = []

    return state

# Assign Relevance Node


# relevance_llm = llm.with_structured_output(RelevanceOutput)

class RelevanceOutput(BaseModel):
    relevance: Literal["H", "M+", "M", "L"]


relevance_parser = PydanticOutputParser(
    pydantic_object=RelevanceOutput
)
def assign_relevance(state: RelevanceState):

    prompt = f"""
You are an experienced patent analyst.

Your task is to assign a relevance rating to the patent.

IMPORTANT:
Use ONLY the relevance framework given below.
Do NOT use your own judgement or criteria.

==================================================
RELEVANCE FRAMEWORK

{state["relevance_framework"]}

==================================================

Product Description

{state["product_description"]}

--------------------------------------------------

Extracted Product Features

{state["product_features"]}

--------------------------------------------------

Extracted Patent Features

{state["patent_features"]}

--------------------------------------------------

Feature Comparison

{state["comparison"]}

--------------------------------------------------

Instructions:

1. Carefully analyse the feature comparison.

2. Follow the relevance framework EXACTLY.

3. Assign ONLY one rating:
   - H
   - M+
   - M
   - L

4. Do not provide any explanation.

Return your answer in the following format:

{relevance_parser.get_format_instructions()}
"""

    response = llm.invoke(
        [
            HumanMessage(content=prompt)
        ]
    )

    try:
        result = relevance_parser.parse(response.content)
        state["relevance"] = result.relevance

    except Exception as e:
        print("Relevance Parsing Error:", e)
        print(response.content)
        state["relevance"] = "L"

    return state

# Generate Rationale Node
class RationaleOutput(BaseModel):
    rationale: str

# rationale_llm = llm.with_structured_output(RationaleOutput)

rationale_parser = PydanticOutputParser(
    pydantic_object=RationaleOutput
)

def generate_rationale(state: RelevanceState):

    prompt = f"""
You are an experienced patent analyst.

A relevance rating has already been assigned.s

Your task is only to provide a concise rationale supporting the assigned relevance rating.

Assigned Relevance Rating

{state["relevance"]}

--------------------------------------------------

Product Description

{state["product_description"]}

--------------------------------------------------

Product Features

{state["product_features"]}

--------------------------------------------------

Patent Features

{state["patent_features"]}

--------------------------------------------------

Feature Comparison

{state["comparison"]}

--------------------------------------------------

Instructions

1. Do NOT change or question the assigned relevance.
2. Base the rationale only on the feature comparison.
3. Mention the most significant matching and/or missing technical features.
4. Keep the rationale concise and professiona and must be exactly in 2-3 sentences.
5. Do not use bullet points or headings.

Return your answer in the following format:

{rationale_parser.get_format_instructions()}
"""

    response = llm.invoke(
        [
            HumanMessage(content=prompt)
        ]
    )

    try:
        result = rationale_parser.parse(response.content)
        state["rationale"] = result.rationale

    except Exception as e:
        print("Rationale Parsing Error:", e)
        print(response.content)
        state["rationale"] = ""

    return state


# Confidence Node

from pydantic import BaseModel, Field
class ConfidenceOutput(BaseModel):
    confidence: int = Field(
        ge=0,
        le=100,
        description="Confidence score between 0 and 100."
    )

confidence_parser = PydanticOutputParser(
    pydantic_object=ConfidenceOutput
)

# confidence_llm = llm.with_structured_output(ConfidenceOutput)

def confidence_node(state: RelevanceState):

    prompt = f"""
You are an experienced patent analyst performing a quality review.

A relevance rating has already been assigned.
Your task is to estimate your confidence in the assigned relevance.

--------------------------------------------------

Assigned Relevance

{state["relevance"]}

--------------------------------------------------

Product Description

{state["product_description"]}

--------------------------------------------------

Product Features

{state["product_features"]}

--------------------------------------------------

Patent Features

{state["patent_features"]}

--------------------------------------------------

Feature Comparison

{state["comparison"]}

--------------------------------------------------

Rationale

{state["rationale"]}

--------------------------------------------------
Instructions

1. Estimate how confident you are that the assigned relevance is correct.

2. Consider:
   - Completeness of feature matching
   - Clarity of the comparison
   - Strength of the technical overlap
   - Presence of ambiguous or uncertain matches

3. Return a confidence score between 0 and 100.

Interpretation:

90–100 : Very High Confidence

75–89 : High Confidence

50–74 : Moderate Confidence

Below 50 : Low Confidence

Return your answer in the following format:

{confidence_parser.get_format_instructions()}
"""

    response = llm.invoke(
        [
            HumanMessage(content=prompt)
        ]
    )

    try:
        result = confidence_parser.parse(response.content)
        state["confidence"] = result.confidence

    except Exception as e:
        print("Confidence Parsing Error:", e)
        print(response.content)
        state["confidence"] = 0

    return state