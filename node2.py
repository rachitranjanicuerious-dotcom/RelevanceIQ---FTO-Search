# This script PydanticOuputParser and has 3 nodes
from langchain_core.messages import HumanMessage
from langgraph.types import interrupt
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import time
from typing import Literal
from state import RelevanceState
from llm import llm

class FeatureExtraction(BaseModel):
    primary_features: list[str]
    secondary_features: list[str]

feature_parser = PydanticOutputParser(
    pydantic_object=FeatureExtraction
)
# Extract_product_features
def extract_product_features(state: RelevanceState):

    prompt = f"""
You are an experienced patent analyst.

Extract only the technical features from the product description.

Classify every extracted feature into either PRIMARY or SECONDARY.

PRIMARY FEATURES:
- Features central to the product's main technical function.
- Features that directly perform the core technical operation.
- Features that are important to the core technical concept.

SECONDARY FEATURES:
- Supporting or auxiliary features.
- Features that are not central to the main technical operation.

Rules
- Do not invent features which are not explicitly mentioned in the description.
- One feature per list item.
- Keep each feature concise.
- Preserve technical terminology.

{feature_parser.get_format_instructions()}
Product Descriptions

{state["product_description"]}

"""
    start = time.time()

    response = llm.invoke(
        [
            HumanMessage(content=prompt)
        ]
    )

    print('Product feature extraction duration:' , time.time() - start)

    print("=" * 80)
    result = feature_parser.parse(response.content)
    state["primary_product_features"] = result.primary_features
    state["secondary_product_features"] = result.secondary_features
    state["review_type"] = "product"
    approved = interrupt(
        {
            "review_type": "product",
            "primary_features": result.primary_features,
            "secondary_features": result.secondary_features
        }
    )
    state["primary_product_features"] = approved["primary_features"]
    state["secondary_product_features"] = approved["secondary_features"]
    return state

# Extract patent features
def extract_patent_features(state: RelevanceState):

    prompt = f"""
You are an experienced patent analyst.

Your task is to extract the important technical features from the
following patent claims.
-Use All Claims as the primary source.
-Publication Number, Title and Abstract are only for context.

Classify every extracted technical feature into PRIMARY or SECONDARY.

PRIMARY PATENT FEATURES:
- Technical features central to the claimed invention.
- Features appearing in the independent claims.
- Features essential to the claimed technical combination.
- Features defining the core technical scope.

SECONDARY PATENT FEATURES:
- Additional technical limitations.
- Features introduced mainly through dependent claims.
- Supporting, monitoring, communication, control, or auxiliary features.

Rules:
- Do not infer features.
- Ignore legal boilerplate.
- Keep each feature concise.
- One feature per list item.

{feature_parser.get_format_instructions()}

Publication Number:
{state["publication_number"]}

Title:
{state["title"]}

Abstract:
{state["abstract"]}

Independent Claim:
{state["independent_claim"]}

All Claims:
{state["all_claims"]}
"""

    start = time.time()
    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )

    print('Patent feature extraction duration:' , time.time() - start)
    try:

        result = feature_parser.parse(response.content)
        print("PATENT PRIMARY FEATURES:")
        print(result.primary_features)
        print("PATENT SECONDARY FEATURES:")
        print(result.secondary_features)

    except Exception as e:

        print("PATENT FEATURE PARSING ERROR:")
        print(e)
        print(response.content)

        state["primary_patent_features"] = []
        state["secondary_patent_features"] = []

        return state

    print("=" * 100)
    print("ABOUT TO INTERRUPT FOR PATENT REVIEW")
    print("PATENT PRIMARY FEATURES:", result.primary_features)
    print("PATENT SECONDARY FEATURES:", result.secondary_features)
    print("=" * 100)

    approved = interrupt(
        {
            "review_type": "patent",
            "publication_number":
                state["publication_number"],

            "primary_features":
                result.primary_features,

            "secondary_features":
                result.secondary_features
        }
    )

    state["primary_patent_features"] = (
        approved["primary_features"]
    )

    state["secondary_patent_features"] = (
        approved["secondary_features"]
    )

    print("APPROVED PRIMARY PATENT FEATURES:")
    print(approved["primary_features"])
    print("APPROVED SECONDARY PATENT FEATURES:")
    print(approved["secondary_features"])
    print("=" * 80)

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

# Confidence 
CONFIDENCE_INSTRUCTIONS = """
# Confidence Assessment

After determining the relevance rating, assess confidence score.

Confidence is certainty in the assigned rating, not relevance strength.

Consider:
1. Is the evidence sufficient to support the assigned rating?
2. Is there a clear alternative rating that could reasonably apply?
3. Are important claim limitations ambiguous?
4. Could another reasonable patent analyst disagree with this rating?
5. Is the distinction between the assigned rating and the closest
   alternative rating clear?

Confidence should be based on these questions, NOT on whether the
rating is H, M+, M, or L.

IMPORTANT:  

Use the full 0–100 scale.

90–100: Strong evidence; little reasonable disagreement.
75–89: Well supported; some uncertainty exists.
50–74: Plausible, but a neighboring rating is reasonably possible.
25–49: Evidence is materially ambiguous or incomplete.
0–24: Insufficient evidence to confidently distinguish the rating.

H, M+, M, and L can all have either high or low confidence.
"""

# Doctrine of Equivalents
Doctrine_of_Equivalents_INSTRUCTIONS = """
Apply the Doctrine of Equivalents.

If a patent feature performs substantially the same function,
in substantially the same way,
to achieve substantially the same result,
treat it as an Equivalent Match even if different terminology is used.
"""

# Rationale Instructions
RATIONALE_INSTRUCTIONS = """

- The rationale must explain WHY the assigned relevance rating is
appropriate based on the technical relationship between the
product and the patent.
- The rationale must be exactly 2–3 sentences.
- The rationale is NOT a restatement of the relevance rating.
- The rationale MUST be based on SPECIFIC FACTS, SPECIFIC FEATURES,
and SPECIFIC CLAIM LIMITATIONS provided in the input.
- Do NOT write a generic explanation based only on the H / M+ / M / L
relevance framework.
- The rationale must identify actual technical features from the
  product and actual technical features or limitations from the patent.
- Prioritize primary features over secondary features.
- Do not invent features, limitations, functions, or relationships.
- If an equivalent relationship is relied upon, explain the technical
  basis for it.
- Explain why the specific facts support the assigned rating rather
  than a neighboring rating.
"""

def final_analysis(state: RelevanceState):
    # ============================================================
    # ANALYSIS 1 — DETAILED MATCH ANALYSIS
    # ============================================================
    detailed_prompt = f"""

You are an experienced patent analyst.
Compare the product features against the patent features.

# RELEVANCE FRAMEWORK
{state["relevance_framework"]}

# PRODUCT DESCRIPTION
{state["product_description"]}

# PRIMARY PRODUCT FEATURES
{state["primary_product_features"]}

# SECONDARY PRODUCT FEATURES
{state["secondary_product_features"]}

--------------------------------------------
# PATENT FEATURES
-------------------------------------------
# PRIMARY PATENT FEATURES
{state["primary_patent_features"]}

# SECONDARY PATENT FEATURES
{state["secondary_patent_features"]}

# TASK
For each product feature, determine whether the patent discloses:

- Exact Match
- Equivalent Match
- Partial Match
- No Match

# Consider:
- Technical function
- Technical effect
- Structural similarity
- Functional equivalence
- Alternative implementations
- Synonymous technical terminology

Do NOT rely only on identical wording.

Apply the {Doctrine_of_Equivalents_INSTRUCTIONS} where technically appropriate.

# Do NOT invent missing features.

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

Use the {CONFIDENCE_INSTRUCTIONS} for determining the confidence of the ratings assigned.

Return:
- comparisons
- relevance
- rationale
- confidence

Use the {RATIONALE_INSTRUCTIONS}
-----------------------------------------------------------------

Return ONLY valid JSON.

{detailed_parser.get_format_instructions()}
"""

    # ANALYSIS 2 — FRAMEWORK ONLY
    # ============================================================

    framework_prompt = f"""

You are an experienced patent analyst.
Perform a independent patent relevance assessment
Use the provided H/M+/M/L framework.

Use the product and patent technical features as evidence.
Do not use the detailed feature comparison analysis to determine
the rating or rationale.

# RELEVANCE FRAMEWORK
{state["relevance_framework"]}

# PRODUCT DESCRIPTION
{state["product_description"]}

# PRIMARY PRODUCT FEATURES
{state["primary_product_features"]}

# SECONDARY PRODUCT FEATURES
{state["secondary_product_features"]}

# PRIMARY PATENT FEATURES
{state["primary_patent_features"]}

# SECONDARY PATENT FEATURES
{state["secondary_patent_features"]}


# IMPORTANT
This is a FRAMEWORK-ONLY analysis.

Do NOT perform:
- Exact Match classification, equivalent match, partial match or no match classification

Do NOT create a comparisons list.

Evaluate the overall relationship between the product features
and patent disclosure.

Then apply ONLY the provided H / M+ / M / L revelance framework.

{CONFIDENCE_INSTRUCTIONS}

Return:
- relevance_framework_only
- rationale_framework_only
- confidence_framework_only

Use the {RATIONALE_INSTRUCTIONS} for rationale_framework_only

Do NOT mention Exact Match, Equivalent Match, Partial Match,
or No Match in the framework-only rationale.

Return ONLY valid JSON.
{framework_parser.get_format_instructions()}
"""
    # ============================================================
    # CALL 1
    # ============================================================

    try:

        start = time.time()

        detailed_response = llm.invoke(
            [
                HumanMessage(content=detailed_prompt)
            ]
        )

        print('Detailed analysis duration:' , time.time() - start)

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
        state["relevance"] = "error"
        state["rationale"] = "error"
        state["confidence"] = 0
    # ============================================================
    # CALL 2
    # ============================================================

    try:

        start = time.time()

        framework_response = llm.invoke(
            [
                HumanMessage(content=framework_prompt)
            ]
        )

        print('Framework only duration:' , time.time() - start)

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
        state["relevance_framework_only"] = "error"
        state["rationale_framework_only"] = ""
        state["confidence_framework_only"] = 0


    # RETURN BOTH ANALYSES
    return state