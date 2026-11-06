# Shared parameters across Systems — design note

*Status: ADOPTED as design direction (Alex, 2026-10-01); not built. Implementation
waits for the close of the R-AIF arc. It amends `amortization.md` A11 refinement (ii) once
built; the current text is quoted below so the delta is explicit. S2 is resolved: gradients
from every reader are accumulated before one update, and alternation is not supported.*

## The capability

Two Systems may declare that they contain the **same mechanism**: one parameter set,
read by both, learned from both. In a physical / Bayesian-mechanical reading this is an
identity claim between parts of two models. The two subsystems are not each carrying a
copy of something similar. There is one quantity, and both are evidence about it. Under
gradient learning that means the shared parameter's gradient is the **sum** of the
gradients of every loss that reads it, the log-loss analogue of multiplying the
likelihoods of two evidence streams about one variable.
*2026-11-06: an identity claim of this kind is a binding of `merge`, which makes the two Systems one
particle (`nesting-and-coupling.md` N2); between particles joined by `couple` a parameter is never
shared, only reported as counts through a likelihood (N6).*

The first concrete customer is R-AIF's preference model. Their CRSPP reads the world
model's encoder, learned initial state and prior head, and its contrast loss trains all
three. Ours keeps its own initial and takes the embedding as data
(`amortization-debt-ledger.md` entry 5). The actor-entropy diagnostic §28–§29 found
this is **not** what separated the two sides' behaviour, so sharing is a modelling
capability and not a fidelity requirement.

## What the rule says today

`amortization.md` A11, refinement (ii): *"module-object sharing across entries RAISES
(the pytree-forking trap: apparent sharing silently duplicates parameters at the first
optimizer step; shared torsos are E's feature by name)."* It is enforced at three points:
the recognition slot (`inference/amortized.py`, the `shared` owner map), the scoring
slot (`objectives/amortized.py`, the `owners` map), and the CRSPP's own validation
(`reference/raif/crspp.py`).

The trap is technical, not semantic. An optimizer step maps a pytree to a new pytree
leaf by leaf, so two references to one Python object become two leaves, each receiving
only its own path's gradient, and they fork at the first update. The rule already
anticipates sharing **by name**. This note proposes delivering that now for parameters
across Systems, independently of the generic q-graph layer.

## Proposal

1. **Declared by address, never by object.** A shared parameter has exactly one
   *home*: an address in one System's `SystemParams` (the same addresses `lr_by_slot`
   and freezing already use, namespaced through `merge()`). Every other reader holds a
   reference token naming that address. Object aliasing keeps raising, with the
   message pointing to the declaration.
2. **Resolution is pure.** References are resolved to the home's arrays at the top of
   any function that reads them, before tracing reaches the forward pass. The pytree
   the optimizer sees contains the parameter once.
3. **One home, one step.** The home address owns the only optimizer state. A trainer
   that steps a shared parameter must see every loss that reads it with gradient, and
   the gradients sum. A reader that is not in the step reads the parameter under a
   stop-gradient, declared per reader (`grad=False`), which is what today's
   external-input path already does for data.
4. **Beliefs are never shared.** Per-tick carries stay per System, as they are now.
   State crosses between Systems as data, through the existing external-input parent
   kind and the `aux_beliefs` seam.
   *2026-11-06: under `nesting-and-coupling.md` N1 a belief is internal and crosses only as a value:
   inside one particle as a sibling-belief read, a modality of the reading component whose datum is
   the source's belief realised by `sample`, `mode` or `stats` (N3); between coupled particles as an
   action node realised from the belief (N5).*
5. **Serialization and description.** A reference is encoded as its home address
   (`description/__init__.py`), never duplicated, so a saved model round-trips with the identity
   intact.

## A related obligation this does not cover

§29's defect was not about parameters. Two modules read one replay batch and disagreed
on its **segmentation**: the world model restarted its filter at `is_first`, and the
preference model carried across episode boundaries. Sharing parameters does not
protect against that. Any System that filters sequences drawn from a shared batch must
receive the batch's episode boundaries. That is the subject of a separate, smaller
follow-up: a contract that a scoring row's System running a filter over batch windows
takes `is_first`, with one test.

## Decisions for Alex

- **S1 — amend A11 (ii).** From "sharing raises" to "aliasing raises; sharing is
  declared by address." Recommendation: yes. The current rule already names
  by-address sharing as the intended feature.
- **S2 — joint step as the default.** Should a shared parameter require a joint step
  over all reading losses, with stop-gradient readers as the explicit exception?
  Recommendation: yes. Today each trainer steps a disjoint set of slots from one
  `LearnState`, which is what makes their call order safe. A shared slot would be
  stepped by two trainers. Alternating two optimizers over it is a legitimate
  algorithm, but it does not learn one quantity from two evidence streams: each
  optimizer's moments see half the gradient stream, the update is order-dependent,
  and the slot's learning rate and clip become ambiguous. **Resolved (Alex):**
  accumulate first, then update; alternation is not supported, and a second trainer
  stepping a shared slot raises. The joint form
  is gradient accumulation by address: each trainer contributes its gradient for the
  shared slot within one train step, and the home applies one update. R-AIF's own
  `model_opt` is this form, since it steps the world model and the CRSPP on their
  summed loss with one Adam.
- **S2′ — sharing as a binding in `merge()`.** `merge()` already yields one
  `SystemParams` and one `LearnState` with namespaced, disjoint addresses. A parameter
  binding (`a.x` is `b.x`), declared like a node binding, would make sharing a
  property of the composite. Within one trainer over the composite the summed gradient
  then comes for free. Recommendation: yes, as the declaration site. It does not
  cover a scoring-slot System such as the CRSPP, which `merge()` refuses, so the
  address-reference form above is still needed.
  *2026-11-06: `nesting-and-coupling.md` N2 classes this binding as an identification: the
  components share the variable outright, with no blanket between them on it.*
- **S3 — scope of the first build.** Recommendation: parameters across Systems
  composed by `merge()` or nested in a scoring slot. Sharing across members of a
  `vmap` population, meaning tied weights across agents, is a different question and
  stays out of scope.
  *2026-11-06: 'nested in a scoring slot' means a System held inside a scoring entry, not a nested
  component of a merge in `nesting-and-coupling.md` N2's sense.*
- **S4 — R-AIF.** Should the reference tier adopt sharing to match their CRSPP exactly
  (ledger rows 4.1 and 4.14), once the capability exists? Recommendation: run it as an
  experiment, not as a fidelity fix. §28–§29 show it is not needed for parity.

## Amendment 2026-11-05: the home is a named parameter variable

*Status: ADOPTED (Alex, 2026-11-05); build starts now, before the showcase's slow tier, which
is its first customer. The first build covers tabular homes (point tables and Dirichlet-over-table);
gradient homes follow on the same declaration.*

### The delta

Proposal 1 above puts a shared parameter's home at one reader's address (`rover.terrain` owns the
table, `mission.map` borrows it). That makes one reader special and leaves the document saying
the table is a property of that factor. The amendment moves the home out of the readers: a
parameter is a **named variable** with its own entry, and every factor that reads it, the first
one included, holds a reference. The factor graph then says what is true: one variable node, two
or more factor nodes attached, and in the description document one record for the variable with
each reader pointing at it. Proposals 2 to 5 stand unchanged with "home" read as the named entry;
the S2 ruling (accumulate every reader's evidence, update once) stands.

### The end state this is a step toward

Parameter variables on a par with state, observation and control variables: declared as nodes
with a family (a Dirichlet over a table, a Gaussian over a matrix), a prior and a belief held like
any node's, edges to the factors that read them, listed as nodes in the description document. In
that engine "factor f reads variable V as its table" is one mechanism whatever V's transition is:
identity with a conjugate update (a parameter as we use the word), or a transition with an action
parent (a controlled table, today's queued cross-factor conditioning item). This step builds the
name, the home and the edge, for homes whose belief is static within an episode and updated by
`learn`. The name and the edge are the invariant; later steps generalise the home.

### What this step builds

- **Declaration.** `SystemParams.parameters`, a mapping from the variable's name to its
  declaration (the same classes the `transition` and `likelihood` groups take). A factor's
  `transition` entry, or a modality's `likelihood` entry, may be `Parameter("<name>")`, a reference
  token, in place of a table. A factor that holds a table rather than a reference owns an implicit
  parameter variable named after itself, so nothing existing changes its spelling; the explicit
  form costs syntax only where a parameter is shared or wants its own name.
- **Resolution.** Pure and at the top of every face, in the step that builds the positional
  view of `SystemParams`: each reference is replaced by the home's declaration, so the internals
  index tables as they do now and the pytree the optimiser and the count update see holds the home
  once. Object aliasing (one Python object under two keys) keeps raising as A11 says; a reference
  is the declared form.
- **Validation.** The home exists and is a table-shaped declaration of the reader's kind
  (child-first, one parent axis per declared parent); every reader's arity and parent sizes are
  checked against the home exactly as they would be against its own table. A rate keyed by a
  referencing reader's slot (`transition:<f>`) raises and names the home's key
  (`parameter:<name>`), which is the one slot the home has.
- **Learning.** Conjugate: the count statistics of every reader with a nonzero rate are summed
  and the home absorbs them in one update; a reader's novelty term (`transition:<f>` in
  `epistemic`) is that reader's expected information about the home from its own predicted
  counts, and selecting it on two readers adds the two expectations. Gradient homes (second
  build): one optimiser state per home, the gradients of every reading loss summed (S2).
- **Pricing.** The description length prices a home once, under `parameter:<name>`; a referencing
  reader contributes its structure (parents, arity) and no table.
- **Description document.** A top-level `parameters` entry records each explicit home (family,
  shape, prior) once; a referencing reader's `transition` or `likelihood` record says
  `"parameter": "<name>"` with its own parents in place of a family. Implicit homes keep today's
  record inside the reader. The page's graph draws a reference as an edge from the variable to the
  reader.
- **Composition.** `merge` namespaces parameter names as it does node names. A binding
  `SharedParameter("<a>.<name>", "<b>.<name>")` identifies two members' parameter variables as one,
  the existing node-identification operation on a fourth node kind: the composite keeps one home
  under the first name and the second member's readers reference it; shapes and families must
  match. `subsystem` views carry the homes their readers reference, and a view's learned
  parameters write back into the composite's `parameters` by name, so two views of one composite
  learn one table.
  *2026-11-06: under `run`, the composite form is to write a learning view back each round and to
  re-take the other views from the composite, at the build, so the example will need no hand-written
  write-back (`nesting-and-coupling.md` N4 and its amendment A3).*

### What stays deferred, and why it is not foreclosed

Per-tick inference about a parameter (a home whose belief steps, which is what a controlled table
needs); readers other than table slots (a parameter as a parent of an arbitrary likelihood, with
its own expectation rule); parameter ports in `couple`, so a separate mind can take another's
Dirichlet as a soft observation; promoting the document's `parameters` entry from a group to the
node list. Each generalises the home or the reader and keeps the name and the reference.
*2026-11-06: the third item is settled by `nesting-and-coupling.md` N6: under `couple` a parameter
crosses only as a report of counts, an action node of its variable realised by `stats` (the counts
since the last report), observed through a multinomial likelihood over the reader's own parameter
variable; `mode` and `sample` of a parameter stay deferred, and nothing clamps a reader's parameter
from outside. The second item is pulled in for that count case only; beyond it, it stays deferred.*

### First customer

The showcase's two tiers: the rover's terrain factor and the scheduler's map factor read one
field (`docs/design/terrain-field.md`, "The slow tier"). Only the rover counts into it; the
scheduler's novelty is honoured by the rover's counting on the fast clock.
*2026-11-06: in `nesting-and-coupling.md` N2's terms the two tiers are a nesting with this one
identification; N6 keeps `SharedParameter` as bookkeeping of one home inside one particle, not a
crossing.*
