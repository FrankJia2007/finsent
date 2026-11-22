# FinSent Gold Set — Annotation Guidelines

Version 1.0 · written 2026-11-22, **before** any gold-set labeling began.

## 0. Why these guidelines exist

The gold set measures **domain shift**: how models trained or prompted on Financial PhraseBank (Finnish
news sentences, 2004–2013) perform on English financial headlines from 2026. For that measurement to be
clean, the *only* thing that may change is the data — **the labeling convention must stay the same as
PhraseBank's**. Otherwise a drop in score mixes two effects (new data vs. new convention) that cannot be
separated afterwards.

The base is therefore the published PhraseBank annotation instruction (Section 1), quoted verbatim.
Section 2 adds operational rules for the cases where that instruction does not decide. Each rule is
derived from the Phase 5 error analysis: the 16 test sentences that every model got wrong
(`outputs/error_hard_core.csv`) all have 50–74% annotator agreement, i.e. the original 16 annotators,
working from the same instruction, disagreed on exactly these cases.

## 1. Base instruction — Malo et al. (2014), verbatim

Source: `data/raw/FinancialPhraseBank-v1.0/README.txt`; Malo, Sinha, Takala, Korhonen & Wallenius,
"Good debt or bad debt: Detecting semantic orientations in economic texts", *JASIST* (2014).

> The objective of the phrase level annotation task was to classify each example sentence into a
> positive, negative or neutral category by considering only the information explicitly available in
> the given sentence. Since the study is focused only on financial and economic domains, the annotators
> were asked to consider the sentences from the view point of an investor only; i.e. whether the news
> may have positive, negative or neutral influence on the stock price. As a result, sentences which have
> a sentiment that is not relevant from an economic or financial perspective are considered neutral.

Three principles follow from it, referred to below as **P1–P3**:

| | Principle |
|---|---|
| **P1** | Use only the information explicitly available in the headline. No outside knowledge, no guessing at context. |
| **P2** | Take an investor's view: may this news have a positive, negative or neutral influence on the stock price? |
| **P3** | Sentiment that is not economically or financially relevant is neutral. |

Labels: **positive**, **negative**, **neutral**. Canonical example from the source:
"The operating margin came down to 2.4 % from 5.7 %." → **negative**.

## 2. Operational rules (this project's interpretation)

These rules are **this project's reading** of P1–P3, not part of the original instruction. They apply
only where P1–P3 leave the label open.

### A1 — Whose stock price?
"The stock price" means the stock of the **subject company** — the company the headline is mainly about
(normally the grammatical subject). If several companies are affected in opposite directions, label for
the subject company only. *(P2)*

### A2 — Procedural and administrative disclosures → neutral
AGM or board approvals of routine matters, filing notices, meeting or publication dates, lending or
custody arrangements, management incentive plans, statements of long-standing relationships. They carry
no explicit information about value. *(P1, P3)*

### A3 — Favorable language without numbers *(decision: option (a), stay close to PhraseBank)*
- **positive** — the headline reports a **concrete favorable development** for the subject company: a new
  investment, a growth initiative, a new contract or customer, an expansion, a product or operational
  improvement, a favorable forecast with a stated figure.
- **neutral** — the language is **generic**: descriptions or praise of a product or location, strategy
  slogans, and the company's own evaluative claims ("strengthens our position", "excellent progress")
  that are not tied to a concrete development.

The mirror rule applies to unfavorable language (concrete setback → negative; generic gloom → neutral).
*(P2 sets a low bar — "may have influence" — so concrete developments count; P3 keeps generic language neutral.)*

### A4 — Events whose direction depends on an unstated reason
Layoffs, divestments, restructurings, management changes: if the headline **states the reason or the
consequence**, label by it ("cuts 200 jobs as demand slumps" → negative; "cuts costs, lifting margin
to 12 %" → positive). If it states neither → **neutral**. Do not infer a reason. *(P1)*

### A5 — Explicit price moves and analyst actions (headline-specific)
Headlines that report the subject company's own share price move ("shares jump 8 %"), an upgrade or
downgrade, or a target-price change are labeled by direction. They state the influence on the stock
price explicitly. *(P1, P2)*

### A6 — Mixed signals (headline-specific)
If the headline contains signals in both directions, label the one the headline presents as the main
news (usually the first clause or the one with a number). If they are genuinely balanced → **neutral**,
and mark confidence `unsure`.

> A5 and A6 are new: PhraseBank sentences rarely contain price moves or "X but Y" constructions, while
> news headlines often do. Revisit both after the pilot batch (Section 4).

## 3. Worked examples — the 16 hard-core PhraseBank sentences under these rules

`PhraseBank` = the original majority label; `agree` = annotator agreement level; `Ours` = the label
under Sections 1–2. Rows where they differ are marked **≠**. (Sentences abridged.)

| # | agree | Sentence | PhraseBank | Ours | Rule |
|---|---|---|---|---|---|
| 0 | 75% | Employees are better prepared to answer calls, since they already have detailed information about the caller | positive | positive | A3 operational improvement |
| 1 | 75% | Olvi has an ongoing MMX Plus project that aims to find growth outside Finland | positive | positive | A3 growth initiative |
| 3 | 66% | One of the installed elevators will be a double-deck elevator, which enables more efficient use of the core space | positive | positive | A3 product improvement |
| 4 | 66% | Nokia will continue to invest in future development of Qt | positive | positive | A3 investment |
| 5 | 66% | Developers predict Skanssi will attract some 4.5 million visitors annually | positive | positive | A3 forecast with figure |
| 8 | 66% | Simdax will remain available in markets where it has been approved (40+ countries) | positive | positive | A3 continued sales |
| 6 | 66% | The core of Solidium's investment strategy is proper, value enhancing asset management | positive | neutral **≠** | A3 strategy slogan |
| 15 | 50% | The apartment block will be well-located, close to a school and kindergarten | positive | neutral **≠** | A3 generic description |
| 10 | 50% | "The sale of the oxygen measurement business strengthens our goal to focus on our chosen segments" | neutral | neutral | A3 company's own evaluative claim |
| 7 | 66% | Nordea Finland and Stockmann have been cooperating for more than ten years | positive | neutral **≠** | A2 long-standing relationship |
| 12 | 50% | The terms of the financing were approved by the AGM | positive | neutral **≠** | A2 approval |
| 13 | 50% | Novator Finland Oy has entered into a securities lending agreement (May 23–26, 2008) | positive | neutral **≠** | A2 lending arrangement |
| 14 | 50% | Panostaja's Board decided on a new long-term incentive plan for management | positive | neutral **≠** | A2 incentive plan |
| 2 | 66% | As a result, the number of personnel in Finland will be reduced by 158 | neutral | neutral | A4 no reason stated in this sentence |
| 11 | 50% | Under the contract, Ericsson will more than double the number of radio base stations in the Elisa network | neutral | positive **≠** | A1 subject = Ericsson, new business |
| 9 | 50% | A tinyurl link takes users to a scam site promising earnings as a Google Cash advertiser | negative | neutral **≠** | P3 no economic relevance to Google's value |

8 of 16 differ from PhraseBank. That is expected: every one of these sentences had 50–75% agreement, so
the PhraseBank label is itself a narrow majority, and the A2 cases (#7, #12–14) are labels that P1–P3
do not support. **The gold set will not be scored against these 16 sentences** — they illustrate the
rules only.

## 4. Procedure

1. **Label blind.** Do not look at any model's prediction (Haiku, Opus, FinBERT, encoders) for a headline
   before labeling it. The gold set is used to choose the Phase 7 teacher; labeling with an LLM's answer
   in view would bias that choice toward the LLM.
2. **Pilot first.** Label ~50 headlines, then reread Sections 2–3 and decide whether A5/A6 or any rule
   needs a change. If a rule changes, relabel the pilot and bump the version at the top of this file.
   Freeze the guidelines before labeling the rest.
3. **Record confidence.** Every label gets `sure` or `unsure`. With a single annotator there are no
   agreement levels; `unsure` is the substitute that lets results be split the way Phase 5 split
   PhraseBank by agreement.
4. **Measure your own consistency.** About one week after finishing, relabel a random 50 (shuffled,
   without looking at the first labels) into `relabel.csv`. Intra-annotator agreement is the noise
   floor of the gold set: model accuracy above it is not meaningful.
5. **Write down hard cases.** When no rule decides a headline, label it, mark `unsure`, and add a short
   note. Recurring notes become new rules in the next guideline version.

## 5. File format

`data/gold/gold.csv`, one row per headline:

| column | meaning |
|---|---|
| `id` | stable integer id |
| `headline` | text exactly as published |
| `source`, `published_at`, `url` | provenance |
| `company` | subject company (A1), ticker if known |
| `label` | `positive` / `negative` / `neutral` |
| `confidence` | `sure` / `unsure` |
| `rule` | rule that decided it, if any (`A2`, `A4`, …) |
| `note` | free text for hard cases |
| `guideline_version` | version of this file used |
| `labeled_at` | date |

`data/gold/relabel.csv`: `id`, `label`, `confidence`, `labeled_at` for the consistency check.
