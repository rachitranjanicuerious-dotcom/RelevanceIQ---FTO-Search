# This has llm with structured output parser and has 6 nodes
import ast
import json
from langchain_core.messages import HumanMessage
from state import RelevanceState
from llm import llm
from pydantic import BaseModel
from typing import Literal

class FeatureExtraction(BaseModel):
    features: list[str]

feature_llm = llm.with_structured_output(FeatureExtraction)

# Extract_product_features
def extract_product_features(state: RelevanceState):

    prompt = f"""
You are an experienced patent analyst.

Your task is to extract the important technical features from the
following product description.

Instructions:

- Extract only technical features.
- Keep every feature short and precise.
- Do not combine multiple features into one.
- Preserve important technical terminology.
- Do not infer or invent features that are not explicitly mentioned.

Product Description:

{state["product_description"]}
"""

    result = feature_llm.invoke(
        [
            HumanMessage(content=prompt)
        ]
    )

    state["product_features"] = result.features

    return state
# ==========================
# Structured Output Model
# ==========================

class FeatureExtraction(BaseModel):
    features: list[str]


structured_feature_llm = llm.with_structured_output(FeatureExtraction)

# Extract patent features
def extract_patent_features(state: RelevanceState):
    """
    Extract technical features from the patent's All Claims.
    """

    prompt = f"""
You are an experienced patent analyst.

Your task is to extract the important technical features from the
following patent claims.

Instructions:

- Use the ALL CLAIMS as the primary source.
- The Independent Claim, Title and Abstract are only for context.
- Extract only technical features.
- Ignore legal language.
- Do not infer features that are not explicitly disclosed.
- Keep each feature concise.
- One feature per list item.


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
    result = structured_feature_llm.invoke(
        [
            HumanMessage(content=prompt)
        ]
    )
    state["patent_features"] = result.features
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


comparison_llm = llm.with_structured_output(ComparisonOutput)

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
- Ignore legal wording.
- Do not infer features that are not present.
- Keep the reasoning to one concise sentence.

Product Features:

{state["product_features"]}

Patent Features:

{state["patent_features"]}

Return the comparison for EVERY product feature.
"""

    result = comparison_llm.invoke(
        [
            HumanMessage(content=prompt)
        ]
    )

    state["comparison"] = result.comparisons

    return state    

# Assign Relevance Node

class RelevanceOutput(BaseModel):
    relevance: Literal["H", "M+", "M", "L"]

relevance_llm = llm.with_structured_output(RelevanceOutput)

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

Return only the relevance rating.
"""

    result = relevance_llm.invoke(
        [
            HumanMessage(content=prompt)
        ]
    )

    state["relevance"] = result.relevance

    return state


# Generate Rationale Node
class RationaleOutput(BaseModel):
    rationale: str


rationale_llm = llm.with_structured_output(RationaleOutput)

def generate_rationale(state: RelevanceState):

    prompt = f"""
You are an experienced patent analyst.

A relevance rating has already been assigned.

Your task is NOT to determine the relevance again.

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
4. Keep the rationale concise and professional.
5. The rationale MUST be exactly 2–3 sentences.
6. Do not use bullet points or headings.

Return only the rationale.
"""

    result = rationale_llm.invoke(
        [
            HumanMessage(content=prompt)
        ]
    )

    state["rationale"] = result.rationale

    return state


# Confidence Node

from pydantic import BaseModel, Field
class ConfidenceOutput(BaseModel):
    confidence: int = Field(
        ge=0,
        le=100,
        description="Confidence score between 0 and 100."
    )


confidence_llm = llm.with_structured_output(ConfidenceOutput)

def confidence_node(state: RelevanceState):

    prompt = f"""
You are an experienced patent analyst performing a quality review.

A relevance rating has already been assigned.

Your task is NOT to assign a new relevance rating.

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

Return only the confidence score.
"""

    result = confidence_llm.invoke(
        [
            HumanMessage(content=prompt)
        ]
    )

    state["confidence"] = result.confidence

    return state
