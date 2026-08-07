# This script PydanticOuputParser and has 3 nodes

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt
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
    # return state

    state["review_type"] = "product" # Added review type

    # Added approved

    approved = interrupt(
        {
            "review_type": "product",
            "features": result.features
        }
    )
    state["product_features"] = approved
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

- Use All Claims as the primary sources.
- Publication Number, Title and Abstract are only for context.
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

    except Exception as e:

        print(e)
        print(response.content)

        state["patent_features"] = []

        return state
    
        # state["patent_features"] = result.features

        # state["review_type"] = "patent" # Added review type

        # Added approved

    approved = interrupt(
        {
            "review_type": "patent",
            "publication_number": state["publication_number"],
            "features": result.features
        }
    )
    
    state["patent_features"] = approved
    return state

# Compare Features
class FeatureMatch(BaseModel):
    product_feature: str
    patent_feature: str
    match_type: Literal[
        "Exact Match",
        "Equivalent Match",
        "Partial Match",
        "No Match"
    ]
    reasoning: str

class DetailedAnalysisOutput(BaseModel):
    comparisons: list[FeatureMatch]

    relevance: Literal["H", "M+", "M", "L"]

    rationale: str

    confidence: int = Field(
        ge=0,
        le=100
    )


class FrameworkOnlyOutput(BaseModel):
    relevance_framework_only: Literal["H", "M+", "M", "L"]

    rationale_framework_only: str

    confidence_framework_only: int = Field(
        ge=0,
        le=100
    )

detailed_parser = PydanticOutputParser(
    pydantic_object=DetailedAnalysisOutput
)

framework_parser = PydanticOutputParser(
    pydantic_object=FrameworkOnlyOutput
)

def final_analysis(state: RelevanceState):

    # ============================================================
    # ANALYSIS 1 — DETAILED MATCH ANALYSIS
    # ============================================================

    detailed_prompt = f"""

You are an experienced patent analyst.

Compare the target product features against the patent features.

# RELEVANCE FRAMEWORK

{state["relevance_framework"]}

# PRODUCT DESCRIPTION

{state["product_description"]}

# PRODUCT FEATURES

{state["product_features"]}

# PATENT FEATURES

{state["patent_features"]}

# TASK

For each product feature, determine whether the patent discloses:

- Exact Match
- Equivalent Match
- Partial Match
- No Match

Consider:

- Technical function
- Technical effect
- Structural similarity
- Functional equivalence
- Alternative implementations
- Synonymous technical terminology

Do NOT rely only on identical wording.

Apply the Doctrine of Equivalents where technically appropriate.

Do NOT invent missing features.

Only classify a feature as Equivalent Match when the patent
reasonably discloses an equivalent technical solution.

If there is no technically relevant disclosure, classify it as
No Match.

After completing the feature-level comparison, assign ONE
relevance rating using ONLY the provided relevance framework.

The relevance rating must NOT be determined merely by counting
matches.

Consider:

- Importance of matched features
- Coverage of core/inventive concept
- Missing essential features
- Claim limitations
- Overall technical overlap

Return:

- comparisons
- relevance
- rationale
- confidence

The rationale must be exactly 2–3 sentences.

Confidence must be an integer from 0 to 100.

Return ONLY valid JSON.

{detailed_parser.get_format_instructions()}
"""

    # ============================================================
    # ANALYSIS 2 — FRAMEWORK ONLY
    # ============================================================

    framework_prompt = f"""

You are an experienced patent analyst.

Perform a SEPARATE patent relevance assessment using ONLY the
provided relevance framework.

# RELEVANCE FRAMEWORK

{state["relevance_framework"]}

# PRODUCT DESCRIPTION

{state["product_description"]}

# PRODUCT FEATURES

{state["product_features"]}

# PATENT FEATURES

{state["patent_features"]}

# IMPORTANT

This is a FRAMEWORK-ONLY analysis.

Do NOT perform:

- Exact Match classification
- Equivalent Match classification
- Partial Match classification
- No Match classification

Do NOT create a comparisons list.

Do NOT use the comparison results from another analysis.

Evaluate the overall relationship between the product features
and patent disclosure.

Then apply ONLY the supplied H / M+ / M / L framework.

The framework-only rating must be independent of any
feature-level match counting.

Return:

- relevance_framework_only
- rationale_framework_only
- confidence_framework_only

The rationale must be exactly 2–3 sentences.

Do NOT mention Exact Match, Equivalent Match, Partial Match,
or No Match in the framework-only rationale.

Confidence must be an integer from 0 to 100.

Return ONLY valid JSON.

{framework_parser.get_format_instructions()}
"""

    # ============================================================
    # CALL 1
    # ============================================================

    try:

        detailed_response = llm.invoke(
            [
                HumanMessage(content=detailed_prompt)
            ]
        )

        detailed_result = detailed_parser.parse(
            detailed_response.content
        )

        state["comparison"] = detailed_result.comparisons
        state["relevance"] = detailed_result.relevance
        state["rationale"] = detailed_result.rationale
        state["confidence"] = detailed_result.confidence

    except Exception as e:

        print("=" * 80)
        print("DETAILED ANALYSIS PARSING ERROR")
        print(e)

        try:
            print(detailed_response.content)
        except:
            pass

        state["comparison"] = []
        state["relevance"] = "L"
        state["rationale"] = ""
        state["confidence"] = 0


    # ============================================================
    # CALL 2
    # ============================================================

    try:

        framework_response = llm.invoke(
            [
                HumanMessage(content=framework_prompt)
            ]
        )

        framework_result = framework_parser.parse(
            framework_response.content
        )

        state["relevance_framework_only"] = (
            framework_result.relevance_framework_only
        )

        state["rationale_framework_only"] = (
            framework_result.rationale_framework_only
        )

        state["confidence_framework_only"] = (
            framework_result.confidence_framework_only
        )

    except Exception as e:

        print("=" * 80)
        print("FRAMEWORK-ONLY PARSING ERROR")
        print(e)

        try:
            print(framework_response.content)
        except:
            pass

        state["relevance_framework_only"] = "L"
        state["rationale_framework_only"] = ""
        state["confidence_framework_only"] = 0


    # ============================================================
    # RETURN BOTH ANALYSES
    # ============================================================

    return state