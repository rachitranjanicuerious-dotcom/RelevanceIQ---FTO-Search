from langchain_core.messages import HumanMessage
from langgraph.types import interrupt
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import time
from typing import Literal
from state import RelevanceState
from llm import llm

# 1. PRODUCT FEATURE EXTRACTION
# ============================================================
class FeatureExtraction(BaseModel):
    primary_features: list[str]
    secondary_features: list[str]

feature_parser = PydanticOutputParser(
    pydantic_object=FeatureExtraction
)

def extract_product_features(state: RelevanceState): 

    prompt = f"""
You are an experienced patent analyst performing an FTO
(Freedom-to-Operate) technical analysis.

Extract the technical features explicitly disclosed in the
product description.

Classify every extracted feature into either PRIMARY or SECONDARY.

PRIMARY FEATURES:
- Core technical features and mechanisms.
- Features important for satisfying potential patent claim
  limitations.
- Features that are important to the core technical concept.  

SECONDARY FEATURES:
- Supporting or auxiliary features.
- Features that are not central to the main technical operation.

Rules:
- Extract only explicitly disclosed features.
- Do not infer or invent features.
- Keep features concise and technically specific.
- One feature per list item.

{feature_parser.get_format_instructions()}

PRODUCT DESCRIPTION:
{state["product_description"]}
"""
    # start = time.time()
    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )
    # print(
    #     "Product feature extraction duration:",
    #     # time.time() - start
    # )

    result = feature_parser.parse(response.content)
    state["primary_product_features"] = result.primary_features
    state["secondary_product_features"] = result.secondary_features

    approved = interrupt({
        "review_type": "product",
        "primary_features": result.primary_features,
        "secondary_features": result.secondary_features
    })

    state["primary_product_features"] = approved["primary_features"]
    state["secondary_product_features"] = approved["secondary_features"]
    return state

# 2. PATENT FEATURE EXTRACTION
# ============================================================

def extract_patent_features(state: RelevanceState):
    
    prompt = f"""
You are an experienced patent analyst performing an FTO
(Freedom-to-Operate) technical analysis.

Extract technical features from the patent claims.

SOURCE PRIORITY:
1. Independent Claim — identify core mandatory limitations.
2. All Claims — identify additional dependent-claim limitations.
3. Abstract and Title —  these are for context only.

Classify every extracted technical feature into PRIMARY or SECONDARY.

PRIMARY FEATURES:
- Technical limitations appearing in independent claims.
- Features defining the core claimed combination.

SECONDARY FEATURES:
- Additional technical limitations.
- Limitations introduced by dependent claims.
- Supporting, optional, control, or auxiliary features.

IMPORTANT:
A dependent-claim limitation must NOT be treated as a mandatory
limitation of the independent claim.

Example:
Claim 1 = A + B + C
Claim 2 = Claim 1 + D

D is required for Claim 2, but not for Claim 1.

Rules:
- Do not infer claim limitations.
- Do not convert abstract/specification disclosure into claim
  limitations.
- Keep features concise.
-e feature per list item.

{feature_parser.get_format_instructions()}

PUBLICATION NUMBER:
{state["publication_number"]}

TITLE:
{state["title"]}

ABSTRACT:
{state["abstract"]}

INDEPENDENT CLAIM:
{state["independent_claim"]}

ALL CLAIMS:
{state["all_claims"]}
"""

    # start = time.time()
    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )
    print(
        "Patent feature extraction duration:",
        # time.time() - start
    )
    try:
        result = feature_parser.parse(response.content)
    except Exception as e:
        print("=" * 80)
        print("PATENT FEATURE PARSING ERROR")
        print(e)
        print(response.content)

        state["primary_patent_features"] = []
        state["secondary_patent_features"] = []

        return state

    # approved = interrupt({
    #     "review_type": "patent",
    #     "publication_number": state["publication_number"],
    #     "primary_features": result.primary_features,
    #     "secondary_features": result.secondary_features
    # })
    # state["primary_patent_features"] = approved["primary_features"]
    # state["secondary_patent_features"] = approved["secondary_features"]

    state["primary_patent_features"] = result.primary_features
    state["secondary_patent_features"] = result.secondary_features

    return state

# 3. OUTPUT SCHEMAS
# ===========================================================

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
    relevance: Literal["H", "M+", "L"]
    rationale: str
    confidence: int = Field(ge=0, le=100)

class FrameworkOnlyOutput(BaseModel):
    relevance_framework_only: Literal["H", "M+", "L"]
    rationale_framework_only: str
    confidence_framework_only: int = Field(ge=0, le=100)

detailed_parser = PydanticOutputParser(
    pydantic_object=DetailedAnalysisOutput
)

framework_parser = PydanticOutputParser(
    pydantic_object=FrameworkOnlyOutput
)

# ============================================================
# 4. FTO INSTRUCTIONS
# ============================================================

COMMON_FTO_INSTRUCTIONS = """
You are an experienced patent analyst performing a technical
Freedom-to-Operate (FTO) relevance screening.

The purpose of this analysis is to determine the technical relevance
and potential claim correspondence between the product and the patent claims.

This is a technical screening assessment only. It is NOT a legal
conclusion regarding infringement, validity, enforceability, or finals
Freedom to Operate.

============================================================
CLAIM-CENTRIC FTO ANALYSIS
============================================================

FTO analysis must be based primarily on the PATENT CLAIMS.

Analyze the strongest relevant patent claims individually.

Do not determine relevance from:
- the patent title, the abstract, the general technical field, specification-only disclosure;
- a simple count of matching product features.

The title, abstract, specification, and examples may be used to
understand technical context, but they cannot replace the limitations
of an actual claim.

============================================================
CLAIM AS A COMPLETE COMBINATION
============================================================

Each claim must be evaluated as the complete combination of its
required limitations.

Do not select a few matching limitations from a claim while ignoring
other material limitations.

At the same time, do not automatically treat every difference in
wording, terminology, or implementation as a material missing
limitation.

Determine which limitations materially define the claimed technical
invention.

============================================================
CLAIM DEPENDENCY
============================================================

Preserve the actual claim hierarchy.
An independent claim contains its own mandatory limitations.

A dependent claim includes:
- all limitations of its parent claim; and
- the additional limitations introduced by the dependent claim.

Example:
Claim 1 = A + B + C
Claim 2 = Claim 1 + D

D is mandatory for Claim 2.
D is NOT a mandatory limitation of Claim 1.

Therefore:
- do not transfer dependent-claim limitations to the parent claim;
- do not use a dependent-claim limitation to downgrade an otherwise relevant independent claim;
- evaluate dependent claims separately.

============================================================
NEVER CREATE A HYPOTHETICAL CLAIM
============================================================

Never combine limitations from separate claims to create a claim that
does not actually exist.

For example, do not take:
- A + B from Claim 1;
- C from Claim 3; and
- D from Claim 5

and treat A+B+C+D as one claimed combination.
Every relevance conclusion must be traceable to an actual claim.

============================================================
PRODUCT EVIDENCE CLASSIFICATION
============================================================

For each important claim limitation, classify the product evidence as:

A. CONFIRMED PRESENT
The product description explicitly supports the limitation or clearly
describes a corresponding technical mechanism.

B. NOT DISCLOSED / UNCERTAIN
The product description does not provide enough information to
determine whether the limitation is present.

This includes situations where:
- the product description is silent;
- the implementation is not described;
- terminology differs and correspondence cannot be conclusively
  established; or
- the available product description is incomplete.

C. CONFIRMED ABSENT / INCOMPATIBLE

The product description affirmatively indicates that:
- the product does not have the limitation;
- the product cannot satisfy the limitation; or
- the product uses a materially different technical mechanism that is
  incompatible with the claimed limitation.

CRITICAL RULE:
NOT DISCLOSED / UNCERTAIN is NOT the same as CONFIRMED ABSENT /
INCOMPATIBLE.

Do not infer absence solely because a limitation is not mentioned.

Silence in a product description is evidence of uncertainty, not
affirmative evidence of absence.

============================================================
DO NOT INVENT PRODUCT FEATURES
============================================================

Do not infer or invent product features that are not supported by the
product description.

Therefore:

No evidence of a feature
→ NOT DISCLOSED / UNCERTAIN.

Affirmative evidence contradicting the feature
→ potentially CONFIRMED ABSENT / INCOMPATIBLE.

============================================================
MATERIAL CLAIM LIMITATIONS
============================================================

A claim limitation is material when it materially defines the claimed
technical invention or the required technical combination.

Potentially material limitations include:
- a required technical component;
- a required technical mechanism;
- a required processing step;
- a required control relationship;
- a required sensor or input;
- a required physical arrangement;
- a required data relationship;
- a required technical constraint; or
- another limitation that materially distinguishes the claimed
  technical solution.

Do not automatically treat the following as material differences:
- terminology;
- naming;
- ordinary contextual language;
- field labels;
- routine implementation details;
- unspecified implementation details; or
- additional product features that are not required by the claim.

Materiality must be determined from the actual claim language and the
product's described technical operation.

============================================================
TECHNICAL EQUIVALENCE
============================================================

Different terminology does not automatically mean different technology.

Where appropriate, treat a product feature as technically equivalent
when it performs substantially the same technical function, through
substantially the same technical mechanism, to achieve substantially
the same technical result.

Use technical equivalence carefully.

Do NOT infer equivalence merely because:
- the purpose is similar;
- the products operate in the same field;
- the result is broadly similar; or
- the terminology sounds related.

Technical equivalence is a technical analytical concept only.

Do not make or imply a legal Doctrine of Equivalents or infringement
conclusion.

============================================================
EXTRA PRODUCT FEATURES
============================================================

Additional product features do not reduce relevance merely because
they are not present in the patent claim.

For example:
Claim = A + B + C
Product = A + B + C + D + E

D and E do not by themselves make the claim less relevant.

The analysis should focus on whether the product can correspond to
the claimed technical combination.

============================================================
SPECIFICATION AND ABSTRACT
============================================================

The specification, examples, abstract, and title may be used to
understand context.

If a feature appears only in the specification but not in the claim,
do not treat that feature as a mandatory limitation of that claim.

============================================================
CLAIM-BY-CLAIM ANALYSIS PROCESS
============================================================

For each candidate claim:

Step 1:
Read the complete actual claim.

Step 2:
Identify its mandatory limitations.

Step 3:
Preserve its independent/dependent claim relationship.

Step 4:
Identify its core technical concept.

Step 5:
Map the product's technical functionality against the important
claim limitations.

Step 6:
Classify important product evidence as:

- CONFIRMED PRESENT;
- NOT DISCLOSED / UNCERTAIN; or
- CONFIRMED ABSENT / INCOMPATIBLE.

Step 7:
Determine whether any confirmed absent limitation is material to the
claimed technical invention.

Step 8:
Consider technical equivalence where justified.

Step 9:
Determine the technical relevance of the actual claim.

Step 10:
Compare the strongest relevant patent claims when necessary.

============================================================
DEPENDENT CLAIM CALIBRATION
============================================================

- If an independent claim substantially corresponds to the product,
evaluate that independent claim based on its own limitations.
- Do not downgrade the independent claim merely because a dependent claim
introduces an additional limitation that is absent or uncertain.
- For a dependent claim, however, all parent limitations plus the
dependent limitation must be considered.

============================================================
STRONGEST CLAIM PRINCIPLE
============================================================

Identify the strongest 1–3 patent claims relevant to the product.

The final rating should be based primarily on the strongest meaningful
claim-to-product correspondence rather than an average across
unrelated claims.

A single highly relevant actual claim can be sufficient to make the
patent highly relevant when the applicable rating framework supports
that conclusion.

============================================================
NO FEATURE COUNTING
============================================================

Do not use a numerical feature-match score as the rating mechanism.

For example:
"8 out of 10 features match, therefore H"
is not valid FTO reasoning.

Instead evaluate:
- the patent claims;
- the complete claimed combination;
- the materiality of limitations;
- the core technical concept; and
- the strength of the product-to-claim correspondence.

============================================================
FTO VS LEGAL CONCLUSION
============================================================

This analysis identifies technical relevance and potential claim
correspondence.

Do not conclude that:
- the product infringes;
- the product does not infringe;
- the patent is valid;
- the patent is invalid;
- the patent is enforceable;
- the product is legally free to operate; or
- the patent is legally unenforceable.

Use technical screening language such as:
- technically relevant;
- potentially corresponding;
- substantial technical overlap;
- material distinction;
- limitation not established;
- limitation appears absent based on the available product evidence.

============================================================
CONFIDENCE
============================================================

Confidence represents certainty in the assigned rating, NOT the
strength of relevance.

Consider:

1. Is there sufficient evidence supporting the rating?
2. Is a neighboring rating reasonably possible?
3. Are important claim limitations ambiguous?
4. Is the product description incomplete?
5. Could another reasonable patent analyst disagree?

90–100:
Strong evidence and little reasonable disagreement.

75–89:
Well supported with some uncertainty.

50–74:
Plausible, but a neighboring rating is reasonably possible.

25–49:
Material ambiguity or incomplete evidence.

0–24:
Insufficient evidence to confidently distinguish the rating.

Confidence may be high or low for any rating.

============================================================
RATIONALE
============================================================

The rationale must be exactly 2–3 sentences.
It must be based on specific facts from the product and patent.

The rationale must:
- identify the relevant actual claim or claimed subject matter;
- identify the relevant product functionality;
- explain the technical correspondence;
- identify important material distinctions where supported;
- distinguish uncertainty from confirmed absence;
- explain why the selected rating is preferred over the nearest
  alternative.

Avoid generic statements such as:

"The patent is relevant because it relates to similar technology."

Do not invent facts.
"""

# ============================================================
# 5. FINAL ANALYSIS
# ============================================================

def final_analysis(state: RelevanceState):

    # DETAILED FTO ANALYSIS
    # ========================================================

    detailed_prompt = f"""
You are performing the FTO RELEVANCE ANALYSIS.

Apply the following FTO methodology:
{COMMON_FTO_INSTRUCTIONS}

============================================================
DETAILED RATING CALIBRATION
============================================================

For this analysis, assign exactly one rating:

H
M+
L

Use the following general RelevanceIQ definitions.

------------------------------------------------------------
H — HIGHLY RELEVANT
------------------------------------------------------------

Assign H when at least one actual patent claim is directed to the same
or substantially aligned core technical functionality as the product,
and there is no clearly established material limitation that the
product lacks, cannot satisfy, or is technically incompatible with.

H can still be appropriate when some claim details are:
- not explicitly disclosed;
- uncertain;
- described using different terminology;
- implementation-specific;
- contextual; or
- routine technical details.

Do not require the product description to reproduce every claim detail
before assigning H.

However, do not assign H when the product evidence clearly establishes
that a material limitation required by the relevant claim is absent or
technically incompatible.

------------------------------------------------------------
M+ — RELEVANT
------------------------------------------------------------

Assign M+ when there is substantial technical overlap between the
product and an actual patent claim, but a meaningful material
distinction prevents the correspondence from being appropriately
classified as H.

A material distinction may include an important claim limitation that
the product evidence affirmatively shows is:
- absent;
- impossible for the product to satisfy; or
- dependent on a materially different technical mechanism.

Do NOT assign M+ merely because:
- a limitation is not mentioned;
- the product description is incomplete;
- terminology differs; or
- a routine implementation detail is uncertain.

------------------------------------------------------------
L — LESS RELEVANT
------------------------------------------------------------

Assign L when the patent's patent claims are not meaningfully directed
to the product's core technical functionality.

L is appropriate when:
- the claimed technical concept is fundamentally different;
- important material claim limitations clearly distinguish the claimed
  invention from the product;
- the strongest correspondence is only generic or peripheral;
- the overlap is primarily at the level of field or purpose;
- the relevant subject matter appears only in the specification or
  abstract rather than the claims; or
- the product does not technically correspond to the actual claimed
  combination.

Do not assign L solely because one or more product details are
undisclosed or uncertain.

------------------------------------------------------------
FINAL RATING TEST
------------------------------------------------------------

Before assigning M+ or L, ask:

"Is the downgrade supported by affirmative product evidence of a
material distinction, or am I treating silence as absence?"

If the distinction is based only on silence or incomplete product
information, treat it as uncertainty and reduce confidence rather than
automatically downgrading the relevance.

============================================================
INPUTS
============================================================

PRODUCT DESCRIPTION:
{state["product_description"]}

PRIMARY PRODUCT FEATURES:
{state["primary_product_features"]}

SECONDARY PRODUCT FEATURES:
{state["secondary_product_features"]}

============================================================
PATENT
============================================================

Publication Number:
{state["publication_number"]}

Title:
{state["title"]}

Abstract:
{state["abstract"]}

INDEPENDENT CLAIM:
{state["independent_claim"]}

ALL CLAIMS:
{state["all_claims"]}

PRIMARY PATENT FEATURES:
{state["primary_patent_features"]}

SECONDARY PATENT FEATURES:
{state["secondary_patent_features"]}

============================================================
ANALYSIS TASK
============================================================

Identify the strongest 1–3 actual patent claims relevant to the product.

Analyze those claims according to the shared FTO methodology.

For the strongest relevant claims, identify the important technical
limitations and compare them against the product.

Return comparisons only for important limitations of the strongest
relevant claims.

Each comparison must contain:

- product_feature
- patent_feature
- match_type
- reasoning

MATCH TYPES:

Exact Match:
The product explicitly supports the patent limitation.

Equivalent Match:
The product uses a technically equivalent mechanism, function, and
result despite different terminology or implementation.

Partial Match:
Meaningful technical correspondence exists, but the particular
limitation is not fully established.

No Match:
The available product evidence affirmatively indicates that the
limitation is absent, incompatible, or materially different.

Do not use No Match merely because the product description is silent.

============================================================
OUTPUT
============================================================

Return:

1. comparisons
2. relevance
3. rationale
4. confidence

The relevance must be exactly one of:

H
M+
L

The rationale must contain exactly 2–3 sentences.

Return ONLY valid JSON.

{detailed_parser.get_format_instructions()}
"""
    
    # ========================================================
    # FRAMEWORK-ONLY FTO ANALYSIS
    # =======================================================

    framework_prompt = f"""
You are performing the FRAMEWORK-ONLY FTO RELEVANCE ANALYSIS.

Apply the following FTO methodology:

{COMMON_FTO_INSTRUCTIONS}

============================================================
USER-PROVIDED RELEVANCE FRAMEWORK
============================================================

The user has supplied the following relevance framework:

{state["relevance_framework"]}

After completing the FTO analysis, use this user-provided
framework to calibrate the final H / M+ / L rating.

The user-provided framework determines the meaning and threshold of
H, M+, and L for this analysis.

Do not invent a different rating framework.

The final rating must be exactly one of:

H
M+
L

============================================================
ANALYSIS INDEPENDENCE
============================================================

Perform this analysis independently.

Do NOT use:
- the detailed analysis rating;
- the detailed analysis rationale;
- the detailed analysis confidence; or
- the detailed analysis comparisons.

Do not attempt to agree with or disagree with the detailed analysis.

Analyze the product and patent independently and apply the user's
framework to the result.

============================================================
INPUTS
============================================================

PRODUCT DESCRIPTION:
{state["product_description"]}

PRIMARY PRODUCT FEATURES:
{state["primary_product_features"]}

SECONDARY PRODUCT FEATURES:
{state["secondary_product_features"]}

============================================================
PATENT
============================================================

Publication Number:
{state["publication_number"]}

Title:
{state["title"]}

Abstract:
{state["abstract"]}

INDEPENDENT CLAIM:
{state["independent_claim"]}

ALL CLAIMS:
{state["all_claims"]}

PRIMARY PATENT FEATURES:
{state["primary_patent_features"]}

SECONDARY PATENT FEATURES:
{state["secondary_patent_features"]}

============================================================
ANALYSIS TASK
============================================================

Identify the strongest actual claim or claims relevant to the product.

Perform the complete claim analysis according to the FTO
methodology.

Then apply the USER-PROVIDED RELEVANCE FRAMEWORK to determine the
final rating.

The final rating must be:

H
M+
L

============================================================
RATIONALE
============================================================

Write exactly 2–3 sentences.

The rationale must:
- identify the relevant actual claimed subject matter;
- identify the corresponding product functionality;
- explain the technical relationship;
- apply the user's relevance framework;
- identify the most important material distinction where supported;
- explain why the selected rating is preferred over the nearest
  alternative.

The rationale must be specific to the actual product and patent.

Do not mention:
- Exact Match;
- Equivalent Match;
- Partial Match; or
- No Match.

Do not make a legal infringement, validity, enforceability, or final
FTO conclusion.

============================================================
OUTPUT
============================================================

Return:

1. relevance_framework_only
2. rationale_framework_only
3. confidence_framework_only

Return ONLY valid JSON.

{framework_parser.get_format_instructions()}
"""


    # ========================================================
    # CALL 1 — DETAILED ANALYSIS
    # ========================================================

    try:

        start = time.time()

        detailed_response = llm.invoke(
            [HumanMessage(content=detailed_prompt)]
        )

        print(
            "Detailed analysis duration:",
            time.time() - start
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
        except Exception:
            pass

        state["comparison"] = []
        state["relevance"] = "error"
        state["rationale"] = "error"
        state["confidence"] = 0


    # ========================================================
    # CALL 2 — FRAMEWORK-ONLY ANALYSIS
    # ========================================================

    try:

        start = time.time()

        framework_response = llm.invoke(
            [HumanMessage(content=framework_prompt)]
        )

        print(
            "Framework only duration:",
            time.time() - start
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
        except Exception:
            pass

        state["relevance_framework_only"] = "error"
        state["rationale_framework_only"] = ""
        state["confidence_framework_only"] = 0

    return state