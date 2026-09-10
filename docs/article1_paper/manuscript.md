# Who and What to Teach: Expertise-Aware Distillation under Client Specialization

*A controlled study with a labeled public proxy*

> Working draft v0.1. Authors, venue and final formatting are pending. Numerical results refer to the published v3 closure snapshot of 2026-09-09 21:48 UTC. Editorial notes and the evidence ledger at the end are not manuscript text. Related work is a starting point, not an exhaustive novelty assessment.

## Abstract

One-shot federated distillation combines locally trained models into a global student, but class specialization challenges the assumption that all teachers should contribute equally to every example. We study this problem with a labeled public proxy and a binary client–class expertise mask estimated independently of teacher checkpoint selection. Our analysis separates three decisions: which teachers contribute, whether their outputs are pooled in logit or probability space, and whether their class support is restricted. Across MNIST, Fashion-MNIST and CIFAR-10, six allocation regimes and three seeds, expertise-based probability pooling substantially improves over uniform probability pooling in specialized conditions. On CIFAR-10, the mean accuracy gain is 28.56 percentage points under Dirichlet α=0.1 and 69.29 points under single-class allocation. However, support restriction can improve target accuracy while reducing student accuracy, and the value of distillation depends on the public-label budget. In the CIFAR-10 IID condition, the advantage over proxy-only supervision is 10.88 points with 500 proxy examples, but becomes −3.73 points with 10,000 examples. These results support expertise-aware routing within the studied protocol while identifying limits to support restriction and to the benefit of teacher knowledge when labeled proxy data are plentiful.

## 1. Introduction

One-shot federated learning requires a server to construct a useful global model after local training, without further rounds of client optimization. Distillation offers a way to combine predictions rather than directly average model parameters. The server evaluates an ensemble of local teachers on a proxy dataset and trains a student to match its targets. The resulting model therefore depends not only on the quality of individual teachers, but also on how their potentially incompatible predictions are combined.

Client specialization makes this combination problem particularly consequential. A teacher trained primarily on a subset of classes may contribute useful predictions within that subset while providing unreliable scores elsewhere. Uniform aggregation does not distinguish these roles. Nevertheless, removing every prediction outside a teacher's accredited classes may also discard relationships that are useful for training the student. Selection of contributors and restriction of their outputs are distinct design choices and should be evaluated separately.

We investigate these choices in a setting with a labeled public proxy and an explicit binary expertise mask. The mask records whether a frozen teacher meets a fixed class-conditional accuracy threshold on a private expertise split. For each proxy example, EXPERT selects teachers accredited for its true class and averages their complete probability distributions. We compare this rule with uniform aggregation and a label-informed ORACLE that selects teachers whose prediction is correct for that individual example. We then vary the pooling operator and restrict each selected teacher's probability support.

A useful distillation target is not sufficient evidence that distillation is preferable to direct use of the public labels. Accordingly, our evaluation includes a student trained by cross-entropy on the same proxy and a focal CIFAR-10 study with nested proxy subsets. These comparisons expose a distinction between recovering performance relative to a weak ensemble target and adding value beyond proxy-only supervision.

Our contributions are an explicit separation of contributor selection, pooling and support in a controlled one-shot protocol; an evaluation of class-conditional expertise routing across six client-allocation regimes; and an analysis of its practical limits through support ablations and public-label budgets. We report both accuracy and negative log-likelihood (NLL), retain unfavorable comparisons, and restrict conclusions to the evaluated architectures, temperatures and budgets.

## 2. Related work and scope

Ensemble distillation provides a mechanism for federated model fusion. FedDF trains a central classifier from client outputs on unlabeled data [1]. We use a uniform logit-pooling reference inspired by this mechanism within our fixed one-shot protocol. The label “FedDF-logit” identifies that reference and should not be read as a reproduction of every component or communication schedule of the original FedDF algorithm.

One-shot learning without auxiliary data addresses a different resource constraint. DENSE studies data-free one-shot federated learning [2], whereas our protocol assumes an existing public proxy and uses its labels for expertise routing. This distinction matters when comparing methods: availability of proxy inputs, labels, expertise statistics and client models must be accounted for separately.

**Editorial task before submission:** expand this section with verified literature on heterogeneous ensemble distillation, expert selection and specialist models, including recent methods compatible with the same information budget. The current experiments do not establish state-of-the-art superiority or the novelty of every mathematical component.

## 3. Setting and methods

### 3.1. Data roles and expertise estimation

Let K teachers solve a C-class classification task. Teacher k produces logits z_k(x) in R^C. Each client has three disjoint private subsets: training data, validation data for checkpoint selection, and expertise data for estimating class-conditional accuracy after selection. The public proxy P contains labeled pairs (x,y), is reserved before private allocation, and is disjoint from all client subsets. The official test set is reserved for final student evaluation.

For the selected, frozen teacher, let n^E_{k,c} be the number of expertise examples of class c and â^E_{k,c} its accuracy on those examples. We define

$$
M_{k,c}=\mathbf{1}\left[n^E_{k,c}>0\;\land\;\widehat a^E_{k,c}\geq\tau_d\right].
$$

The threshold τ_d is fixed per dataset. A zero mask entry denotes competence not accredited by this rule, including the case of no observations. It does not establish incompetence. In particular, the statistic measures correct prediction conditional on the true class; it does not certify rejection of inputs from other classes.

### 3.2. Contributor selection and pooling

At temperature T>0, define p_k^T(x)=softmax(z_k(x)/T). For a nonempty selected set S(x), the two pooling operators are

$$
q_{\mathrm{logit}}(x)=\operatorname{softmax}\left(\frac{1}{T|S(x)|}\sum_{k\in S(x)}z_k(x)\right),
\qquad
q_{\mathrm{prob}}(x)=\frac{1}{|S(x)|}\sum_{k\in S(x)}p_k^T(x).
$$

Uniform aggregation uses all teachers. EXPERT uses the proxy label and mask to select

$$S_E(x)=\{k:M_{k,y(x)}=1\}.$$

ORACLE instead selects teachers whose individual top prediction matches the proxy label:

$$S_O(x)=\{k:\arg\max_c z_{k,c}(x)=y(x)\}.$$

ORACLE is a sample-level diagnostic reference rather than a guaranteed upper bound on student performance. Both EXPERT and ORACLE exploit proxy labels, but encode different selection rules. The six-method main study contains uniform and ORACLE pooling in both spaces, EXPERT probability pooling, and its support-restricted variant.

If a selected set is empty, all selection-based variants use the common fallback

$$q_{\mathrm{fallback}}(x)=\operatorname{softmax}\left(\frac{1}{KT}\sum_{k=1}^{K}z_k(x)\right).$$

Thus, the probability-based variants contain an explicit logit-pooling exception on fallback examples. We record these events rather than silently treating them as ordinary probability aggregation.

### 3.3. Support restriction

EXPERT-prob retains each selected teacher's complete distribution. EXPERT-prob-SR uses the same selected set but restricts each teacher separately:

$$
\widetilde p^T_{k,c}(x)=
\frac{M_{k,c}\,p^T_{k,c}(x)}{\sum_j M_{k,j}\,p^T_{k,j}(x)},
\qquad
q_{\mathrm{SR}}(x)=\frac{1}{|S_E(x)|}\sum_{k\in S_E(x)}\widetilde p_k^T(x).
$$

Every selected teacher has nonempty support because it is accredited for y(x). The implementation computes a stable softmax over supported logits, avoiding division by numerically underflowed probability mass. Empty selection still invokes the common fallback.

For a non-fallback example, restriction cannot decrease the probability each selected teacher assigns to the true class: that class remains in support and the normalization denominator is at most one. Consequently, true-class target probability cannot decrease and target NLL cannot increase. This algebraic property does not guarantee improved target top-1 accuracy or improved student generalization. It makes student-level evaluation essential.

### 3.4. Student objectives

The distillation student with logits s_θ(x) minimizes

$$
\mathcal L_{\mathrm{KD}}=T^2\,\mathbb E_{x\in P}\left[
\operatorname{KL}\left(q(x)\,\Vert\,\operatorname{softmax}(s_\theta(x)/T)\right)\right].
$$

The supervised reference minimizes cross-entropy against y(x) on the same proxy subset. Neither reference receives private training examples. Teacher-derived targets do carry information learned from private data, but the CE–KD comparison also changes the training objective and therefore does not isolate a causal effect of private knowledge alone.

## 4. Experimental protocol

We evaluate MNIST, Fashion-MNIST and CIFAR-10 with ten clients and seeds 42, 43 and 44. A balanced proxy of 10,000 examples is reserved from the official training data, leaving 50,000 private examples for MNIST and Fashion-MNIST and 40,000 for CIFAR-10. Client allocations comprise IID, class-wise Dirichlet α∈{1.0,0.5,0.1}, two-class allocation (multi), and single-class allocation. In multi, five disjoint class pairs are each shared by two clients. Dirichlet allocation changes both class composition and client sample counts; these regimes are not a one-dimensional causal intervention on specialization.

Each client's examples are split within class into 70% training, 10% validation and 20% expertise using largest-remainder rounding. Rare classes need not appear in every split. Teachers are selected by validation accuracy with strict improvement, then frozen before expertise estimation and proxy inference. Thresholds are 0.90, 0.80 and 0.70 for MNIST, Fashion-MNIST and CIFAR-10, respectively.

The implementation uses MnistNet for MNIST/Fashion-MNIST and ResNet9 for CIFAR-10. Local teachers use Adam with learning rate 10^-3, batch size 64, a maximum of 50 epochs and early-stopping patience of five. CIFAR teacher training includes random cropping and horizontal flipping; student proxy inputs use deterministic evaluation transforms. Students use AdamW with learning rate 10^-3, weight decay 10^-4, batch size 256 and no learning-rate scheduler. All main distillations use T=8 and 1,200 updates. These settings describe the implemented recipe; execution provenance is retained separately rather than assigning the current commit retrospectively to all runs.

Paired KD comparisons share teachers, proxy, expertise mask where applicable, student initialization, batch order and update budget. CE comparisons share the relevant proxy and student-training controls. We report differences paired by seed, their mean and sample standard deviation. Official-test accuracy and NLL are evaluated using the final student's ordinary logits, without applying the distillation temperature at evaluation. Target metrics describe training targets on the proxy and are not held-out generalization estimates.

The main study comprises 324 KD executions and nine full-proxy supervised executions. The CIFAR budget study uses N∈{100,500,1000,5000,10000} in IID, α=0.1 and single. Smaller proxies are balanced nested subsets of the original proxy. This adds 12 CE and 36 KD executions; with reused full-proxy anchors, the curve contains 60 unique executions and 45 paired comparisons. CE anchors are shared across regimes and do not constitute additional independent replicates. The effective batch is min(256,N), with incomplete final batches retained, so fixing updates does not fix the number of examples consumed.

## 5. Results

### 5.1. Selection becomes consequential under specialization

Expertise routing produces much larger gains over uniform probability pooling in specialized conditions than in IID. In CIFAR-10, EXPERT-prob−FedDF-prob is −0.08 percentage points (pp) in IID, +28.56 pp under α=0.1, +26.87 pp in multi and +69.29 pp in single. The corresponding single-class gains are +88.73 pp for MNIST and +80.30 pp for Fashion-MNIST. These differences support the utility of the labeled selection procedure in the studied regimes, rather than universal superiority of any teacher-ranking rule.

![Selection and expertise](figures/routing_student_test_accuracy.png)

**Figure 1.** Paired accuracy differences across allocation regimes. Top: ORACLE-logit−FedDF-logit. Bottom: EXPERT-prob−FedDF-prob. Markers show means and bars sample SD over three seeds. Each panel uses its own vertical scale. The rows differ in selection information and pooling operator; comparing their magnitudes is not a causal decomposition.

ORACLE does not always outperform EXPERT. In CIFAR-10 α=0.1, ORACLE-prob−EXPERT-prob is −4.06 pp. Correctness-based selection of individual teacher predictions does not directly optimize the final student's generalization. We therefore interpret ORACLE as a diagnostic of alternative routing rather than a ceiling that EXPERT must approach from below.

### 5.2. The effect of pooling depends on selection

Probability and logit pooling yield materially different students for uniform aggregation even at T=8. In CIFAR-10, FedDF-prob improves over FedDF-logit by 12.54 pp under α=0.1 and 11.39 pp in multi. For ORACLE, the corresponding differences are −1.89 pp and −0.05 pp. Pooling effects therefore depend on the selected ensemble and cannot be inferred from a single routing rule.

The bounded coordinates of a probability distribution do not make probability pooling invariant to teacher logit scale: changing that scale still changes softmax confidence. Our findings establish an empirical operator comparison at the evaluated temperature, not a proof that probabilities are universally preferable or that calibration mismatch is eliminated.

### 5.3. Restriction improves some target diagnostics while harming students

Support restriction does not consistently improve the student. On CIFAR-10 it reduces mean accuracy in five of six regimes: SR−full is −6.53 pp in IID, −6.74 pp under α=0.1 and −1.31 pp in multi, while single yields +0.57 pp. In IID, target accuracy improves by 9.53 pp despite the lower student accuracy. This discrepancy demonstrates why a target-level improvement cannot substitute for evaluation of the distilled model.

These observations are compatible with the possibility that complete distributions contain useful information, but they do not identify removal of dark knowledge as the unique cause. Restriction changes target concentration and relationships between classes simultaneously. Moreover, some improvement in true-class target probability is guaranteed by the labeled restriction rule itself.

### 5.4. Public-label supervision changes the practical comparison

At N=10000, EXPERT-prob has lower mean accuracy than CE in every CIFAR-10 regime. Its differences are −3.73 pp in IID, −2.64 pp under α=0.1 and −0.25 pp in single. The result qualifies the large gains over uniform distillation: repairing an ensemble target does not necessarily outperform learning directly from the available labels.

Accuracy and NLL do not always favor the same procedure. In CIFAR-10 IID, EXPERT-prob has lower accuracy than CE but improves NLL by 0.1132. In single, its NLL is worse by 0.8369 despite the small mean accuracy difference. We consequently avoid an undifferentiated claim of better performance and report both metrics.

**Table 1. CIFAR-10 paired effects at N=10000 and T=8 for KD.** Mean ± sample SD over three seeds. Accuracy differences are in pp; lower ΔNLL favors EXPERT over CE. All six regimes are shown. These are paired-effect statistics, not absolute method performance.

| Regime | EXPERT−uniform accuracy (pp) | SR−full accuracy (pp) | EXPERT−CE accuracy (pp) | EXPERT−CE NLL |
|---|---:|---:|---:|---:|
| iid | -0.08 ± 0.39 | -6.53 ± 1.24 | -3.730 ± 0.885 | -0.113 ± 0.054 |
| alpha1p0 | +1.38 ± 0.16 | -2.78 ± 0.63 | -3.457 ± 1.082 | -0.110 ± 0.050 |
| alpha0p5 | +6.54 ± 0.93 | -4.01 ± 1.24 | -3.633 ± 0.798 | -0.085 ± 0.068 |
| alpha0p1 | +28.56 ± 11.18 | -6.74 ± 2.24 | -2.643 ± 1.049 | +0.153 ± 0.107 |
| multi | +26.87 ± 4.15 | -1.31 ± 1.09 | -3.917 ± 1.657 | +0.700 ± 0.015 |
| single | +69.29 ± 0.71 | +0.57 ± 0.87 | -0.253 ± 1.290 | +0.837 ± 0.166 |

### 5.5. The value of distillation depends on the proxy budget

The CIFAR-10 IID curve reveals a substantial advantage for KD at smaller proxy sizes. EXPERT-prob−CE is +10.88 pp at N=500 and +10.41 pp at N=1000, decreasing to +0.93 pp at N=5000 and −3.73 pp at N=10000. Thus, the comparison at the full proxy alone would miss a region where teacher-derived targets are beneficial under the fixed update budget.

The pattern is weaker and more variable under α=0.1, where the mean accuracy differences are +1.23 pp at N=500 and +1.18 pp at N=1000, but −2.36 pp at N=5000. In single, N=1000 gives a +2.93 pp accuracy difference alongside a +0.8606 NLL penalty. Benefits in classification accuracy need not imply improved probabilistic predictions.

![Public-proxy budget](figures/proxy_curve.png)

**Figure 2.** EXPERT-prob−CE on CIFAR-10 as a function of N. Top: accuracy differences in pp; bottom: NLL differences. Colored lines represent seeds and black markers show their mean with sample SD. Positive values favor KD for accuracy and disfavor it for NLL. The 1,200-update budget is shared, but examples consumed need not match across N. Lines connect evaluated sizes and do not estimate a continuous response function.

The changes of sign locate descriptive intervals between sampled sizes, not an optimal N or a statistically established equivalence threshold. The curve's conclusions are restricted to CIFAR-10, the three selected regimes and the fixed training recipe.

## 6. Discussion and limitations

Expertise routing requires information that uniform aggregation does not use. The server has both proxy labels and a competence mask derived from labeled private data. This budget is part of the method, not an implementation detail. A particularly important unresolved comparison is whether estimating class-conditional accuracy adds value beyond a mask recording which classes occur in local training. Our present results compare the complete procedure with uniform pooling but do not isolate this distinction.

The expertise split removes reuse of checkpoint-selection data for estimating M, but it does not make all competence estimates precise. Rare classes can have little evidence, and the selected expertise cells are not independently validated on a local test set. Nor does class-conditional accuracy establish rejection of out-of-support examples. These limitations matter especially for a future extension that must route without the proxy label.

The experimental scope is intentionally controlled. We evaluate three datasets, fixed architectures, three seeds and one main distillation temperature. Client allocations also change sample quantities and support. Incomplete archived controls are excluded from claims requiring a complete comparison. The study is not an exhaustive SOTA benchmark, a demonstration of communication efficiency, or a comparison of optimally tuned CE and KD recipes.

The v3 audit records cache and partition hashes, result identities and paired-run metadata. Checkpoint files were fingerprinted as bytes, but the published review did not independently recompute their tensor-state hashes by deserialization. Several execution commits are recorded. Historical results were known during project development; restarting the protocol does not make the official test previously unseen. We therefore distinguish the fixed recipe used for these comparisons from a claim that the entire research process was preregistered.

## 7. Conclusion

Class-conditional expertise offers a useful selection rule for one-shot distillation with a labeled public proxy. In the evaluated specialized regimes, it substantially improves students over uniform pooling. However, pooling space, support restriction and public-label availability have distinct effects: probability pooling is not interchangeable with logit pooling, more faithful targets can yield worse students, and the advantage over direct supervision depends on proxy size and regime. These findings motivate evaluating both the construction of ensemble targets and the practical value they add beyond available labels. Extending the approach to unlabeled or synthetic proxies requires additional evidence about discrimination outside each teacher's accredited classes.

## References

[1] Tao Lin, Lingjing Kong, Sebastian U. Stich, and Martin Jaggi. *Ensemble Distillation for Robust Model Fusion in Federated Learning*. NeurIPS, 2020. https://proceedings.neurips.cc/paper/2020/hash/18df51b97ccd68128e994804f3eccc87-Abstract.html

[2] *DENSE: Data-Free One-Shot Federated Learning*. NeurIPS, 2022. https://papers.nips.cc/paper/2022/hash/868f2266086530b2c71006ea1908b14a-Abstract-Conference.html

## Editorial evidence ledger — remove before submission

| Draft component | Evidence | Status |
|---|---|---|
| Abstract: expertise gains | main_contrast_summary.csv, expertise_gain/student_test_accuracy | Supported as descriptive paired means |
| Abstract: N=500 and N=10000 | proxy_curve_summary.csv and 45 published pairs | Supported for CIFAR IID only |
| Methods and recipe | partitioning.py, local_training.py, distillation.py, runner.py | Checked against reviewed code; retain execution provenance |
| Section 5.1 and ORACLE gap | main_contrast_summary.csv | Supported; no upper-bound claim |
| Section 5.2 | feddf_pooling and oracle_pooling summaries | Supported at T=8; EXPERT pooling incomplete |
| Section 5.3 | support student and target summaries | Supported; mechanism unproven |
| Section 5.4 | private_knowledge_summary.csv | Supported; objective also differs |
| Section 5.5 | proxy_curve_summary.csv and proxy_curve_paired.csv | Supported; no optimal-N claim |
| Novelty/SOTA positioning | Two starting references only | Needs expanded literature review |
| Absolute method results and main seed-level tables | Full local closure snapshot, not all rows publicly bundled | Export before submission |

### Revision checklist

- Contribution: describe routing and the controlled study; establish novelty against the expanded literature before claiming it.
- Clarity: keep labeled proxy and the common logit fallback explicit; define teacher support separately from contributor selection.
- Experimental strength: add absolute accuracy/NLL tables and SD for all headline numbers; decide whether to complete controls or the training-support-mask comparison.
- Evaluation completeness: retain CE and unfavorable NLL outcomes; do not extrapolate the CIFAR curve to MNIST/Fashion-MNIST.
- Method soundness: audit the information budget and historical test exposure; do not infer OOD expertise from conditional accuracy.

Next writing pass: complete Related Work, add the absolute-result table from the original snapshot, then finalize figure selection, captions and abstract length for the target venue. No new training is authorized or performed by this draft.
