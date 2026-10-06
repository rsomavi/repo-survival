# OSS Pulse: predicting the survival of GitHub repositories

DATA11001 Data Science — University of Helsinki
Rubén Somavilla

---

## 1. Introduction

Most repositories published on GitHub never really take off: they get
created, see some activity at the start, and are then abandoned without
anyone touching them again. This project starts from a specific
question: can we tell, just by looking at a repository's first month of
life, whether it will still be active months or years later?

This is a question with practical value beyond the academic exercise.
Someone deciding whether it's worth investing time in someone else's
project, or a maintainer trying to understand which early signals
distinguish projects that thrive, would benefit from a data-driven
answer rather than intuition.

The project covers the full process: data collection through the GitHub
API, cleaning and feature construction, exploratory analysis, training
and comparing several models, and a web application where the
prediction can be tried out on any public repository.

## 2. Data collection

Data was obtained through the GitHub REST API, restricting the sample to
Python repositories created between 2023 and 2024. This range was chosen
to allow enough time perspective: a repository needs time ahead of it to
be genuinely considered abandoned, not just paused for a few weeks.

The first version of the collection sorted search results by last
update date, aiming for variety rather than pulling only the most
popular repositories. When the target variable was computed on that
initial sample, 100% of repositories came out as "active" — clearly
impossible, and a sign of a problem in the design rather than in the
calculation. Sorting by "most recently updated" selects, almost by
definition, repositories with recent activity, which is exactly the
variable that was meant to be predicted.

The fix was to redesign the collection with stratified sampling: 24
independent queries combining 8 creation quarters with 3 star ranges (0,
1-49, 50+), with no sort criterion related to popularity or activity.
This produced a sample of 2,400 repositories with a much more credible
distribution: 31% active, 69% abandoned.

A second, more technical problem appeared while collecting per-repo
features. The count of open and closed issues relied on a common REST
API trick: requesting a single item per page and reading the `Link`
header to find the last page, in order to infer the total without
downloading every item. This method fails for repositories with many
issues, because GitHub doesn't always include that information when it
is expensive to compute, and the code, unable to find it, defaulted to
assuming there was only one item. Almost every repository in the sample
showed exactly one open and one closed issue — something discovered by
comparing against the `open_issues_count` field the API itself returns
for a known, active repository. The problem was fixed by replacing that
count with a GraphQL query, which returns the exact number without
depending on pagination.

## 3. Cleaning and feature construction

After removing duplicates and a handful of records that failed during
collection, 2,397 repositories remained. A repository is considered
active if it received a commit in the last 6 months, and abandoned
otherwise. This single criterion was chosen over combining activity with
popularity, after the exploratory analysis showed that popularity alone
doesn't separate alive projects from dead ones very well.

Every feature below is computed exclusively from each repository's
**first 30 days of existence** — nothing more recent is used, since the
whole point of the model is to predict from early signals only. The two
exceptions are `days_since_push`, which is only used to build the label
itself (not fed to the model), and `age_days`, which was tested and
ultimately excluded from the final model (see Section 5).

| Feature | What it measures | How it's computed | Why it's included |
|---|---|---|---|
| `stargazers_count` | Stars at the time of collection | Direct field from the GitHub API | Rough proxy for visibility/popularity |
| `forks_count` | Forks at the time of collection | Direct field from the GitHub API | Proxy for reuse/interest from other developers |
| `first_month_commits` | Commit activity in the first 30 days | Counted via the GitHub commits endpoint, filtered by `since`/`until` on the creation date | Direct measure of early development intensity |
| `contributors_count` | Number of distinct people who contributed | Counted via the GitHub contributors endpoint (excluding anonymous contributions) | One of the strongest predictors found in the exploratory analysis — more early contributors correlates with survival |
| `issue_close_ratio` | Share of issues that were closed, out of those opened | `closed_issues / (open_issues + closed_issues)`, counted via a GraphQL query for exact totals; 0 when there are no issues | The single strongest correlate of survival (r = 0.43) — closing issues signals an actively maintained project |
| `commits_per_contributor` | Average commits per person in the first month | `first_month_commits / contributors_count`; 0 when there are no contributors | Distinguishes a project driven by one person from one with distributed effort (see the counter-intuitive finding in Section 4) |
| `has_license` | Whether the repo has a license | Boolean, from the GitHub API's `license` field | Associated with a much higher survival rate (~45% vs ~13%) — a signal of early intent |
| `has_wiki` | Whether the wiki feature is enabled | Boolean, from the GitHub API | Included to test it. Repos with the wiki enabled survive less often (~26% vs ~44%), a counter-intuitive pattern discussed in Section 4; the model barely uses it |
| `has_description` | Whether the repo has a description | Boolean, from the GitHub API | Same kind of early-intent signal as `has_license` (~39% vs ~10%) |
| `age_days` *(excluded)* | Days since creation | `today − created_at` | Initially included, but dropped after it turned out to be structurally tied to how the label itself is defined (see Section 5) |
| `days_since_push` *(label only)* | Days since the last commit | `today − pushed_at` | Used only to build the target variable (`label = 1` if ≤ 180 days), not passed to the model |

These derived variables were combined with the metadata fields above
into the final feature set used for both the exploratory analysis and
model training.

## 4. Exploratory analysis

The analysis is in `notebooks/01_eda.ipynb`; every figure below is saved
by that notebook to `docs/images/`. The dataset has 2,397 repositories,
of which 31% are active and 69% abandoned.

![Class balance](images/class_balance.png)

The imbalance is moderate, so it is handled with stratified splits and
class weights rather than resampling.

Stars, forks and contributor counts follow a power law: half of the
repositories have fewer than 50 stars, while a few reach more than
180,000. On a linear scale almost everything piles up at zero, so these
variables are plotted on a log scale from here on.

![Distribution of stars](images/stars_distribution.png)

### Key findings

**1. Collaboration and issue handling separate the two classes best.**
Active repositories have a median of roughly 12 contributors in their
first month, against 1 for abandoned ones, and a median issue close
ratio of about 0.7 against 0. They also have more first-month commits.
The abandoned class is concentrated at the bottom of all three ranges.

![Feature distributions by label](images/boxplots_by_label.png)

**2. Early signs of intent matter.** Repositories with a license are
active 45% of the time versus 13% without one, and those with a
description 39% versus 10%.

**3. The wiki flag points the opposite way from what one might expect.**
Repositories with the wiki enabled are active about 26% of the time,
and those without it about 44%. It is unlikely that disabling the wiki
keeps a project alive. More plausibly the flag is a proxy for something
else, such as how the repository was set up or by whom, but this was not
investigated further. In the final model it is the least important
feature.

![Survival rate by repository flags](images/survival_by_flags.png)

**4. Correlations are moderate, and popularity is redundant.**
`issue_close_ratio` has the strongest correlation with the label
(0.43), followed by `contributors_count` (0.35). Stars and forks
correlate at 0.88 with each other, so they carry largely the same
information. `commits_per_contributor` has almost no linear correlation
with the label (-0.02), but the boxplots show it is slightly higher in
abandoned repositories: many of them are the work of one person who
commits heavily at the start and then stops, while surviving projects
spread the work across more people.

![Correlation matrix](images/correlation_heatmap.png)

**5. Age is almost uncorrelated with the label (-0.09), and yet it
turned out to matter to a model.** This is discussed in the next
section.

## 5. Modeling

The modeling is in `notebooks/02_modeling.ipynb`. Four model families
were compared on the same stratified 80/20 split (1,917 training and 480
test repositories, with the 31% / 69% class proportions preserved in
both): logistic regression, Random Forest, Gradient Boosting and a
small neural network (MLP). Models that are sensitive to feature scale
(logistic regression and the MLP) include the scaler inside a
scikit-learn `Pipeline`, so it is fitted on training data only.

### A shortcut hidden in the label

Before comparing models, a problem came up. In the first Random Forest
that included repository age, `age_days` came out as the most important
feature, even though its correlation with the label is close to zero.

![Feature importance with age_days](images/feature_importance_with_age.png)

The reason is structural. The label is defined as "pushed within the
last 180 days", so an older repository has had, by the passage of time
alone, more chances to fall into an inactive window. The feature does
not describe the project, it describes how the label was built. With
`age_days` the Random Forest reaches an AUC of 0.851; without it, 0.817.
The drop confirms that part of that score came from the shortcut, and
that the remaining features still carry real signal. `age_days` was
removed and every result below uses the nine remaining features.

### Default hyperparameters

| Model | AUC | Accuracy | Precision (Active) | Recall (Active) |
|---|---|---|---|---|
| Gradient Boosting | 0.823 | 0.802 | 0.701 | 0.631 |
| Random Forest | 0.817 | 0.792 | 0.667 | 0.658 |
| Logistic Regression | 0.800 | 0.694 | 0.505 | 0.745 |
| Neural Network (MLP) | 0.789 | 0.808 | 0.739 | 0.591 |

### Hyperparameter tuning

Each model was tuned with a grid search (logistic regression) or a
randomized search (the other three) using stratified 5-fold
cross-validation on the training set only. The final model was chosen by
cross-validated AUC. The test set was used once per model, only to report
the number, and never to choose anything.

| Model | AUC, default (test) | AUC, tuned (CV) | AUC, tuned (test) |
|---|---|---|---|
| Gradient Boosting | 0.823 | 0.853 | 0.826 |
| Random Forest | 0.817 | 0.850 | 0.835 |
| Neural Network (MLP) | 0.789 | 0.821 | 0.804 |
| Logistic Regression | 0.800 | 0.814 | 0.800 |

Tuning improved the test AUC only slightly (by 0.003 for Gradient
Boosting, 0.018 for Random Forest, 0.015 for the MLP and not at all for
logistic regression), which suggests the limit is in the information
carried by the features rather than in the hyperparameters. Gradient
Boosting and Random Forest are effectively tied: they differ by 0.003 in
cross-validated AUC and by 0.009 in the opposite direction on the test
set, both well within the noise of a test set of 480 rows. Gradient
Boosting was kept because the selection rule, cross-validated AUC, picks
it; choosing Random Forest because it scored higher on the test set
would have meant selecting on the test set.

The final model is a Gradient Boosting classifier with 200 trees, a
learning rate of 0.05, a maximum depth of 2 and a subsample of 0.85. On
the test set it reaches an AUC of 0.826 and an accuracy of 0.82. For the
active class, precision is 0.74 and recall 0.64: when it says a
repository will survive it is right about three times out of four, and
it finds about two thirds of the repositories that really do.

The neural network has the best precision among the default models but
the lowest AUC, which is what one would expect from a dataset of under
2,500 rows that gives it little room to use its capacity. Logistic
regression has the best recall of all, at the cost of many false
positives.

### What the model relies on

![Feature importance of the final model](images/feature_importance_final.png)

Contributor count dominates, with an importance of about 0.58, followed
by issue close ratio at about 0.18. All other features are below 0.06.
This matches the exploratory analysis: the model is mostly measuring
whether a project had other people working on it and whether issues were
being handled. License and description, which looked strong in the
exploratory analysis, add little once contributors and issues are known,
probably because they overlap with those signals.

## 6. Results and limitations

random classifier and over a trivial model that always predicts the
majority class. Even so, it's worth being upfront about what the system
doesn't do well.

The dataset is limited to Python repositories, so there's no guarantee
the same behavior holds for other languages. It also can't capture
repositories their owners deleted outright, since those never appear in
a GitHub search; the model distinguishes between "still exists and
active" and "still exists but inactive," not between survival and
disappearance.

A more interesting limitation surfaced when testing the system on a real
case: the author's own bachelor's thesis repository, developed solo.
The model gave it only a 6-7% probability of being active, despite it
being actively worked on in practice. The reason is that the model's
features — contributors, closed issues, a public license — reflect
patterns typical of collaborative open-source development, and a
personal or academic project that doesn't use those tools publicly
shows the same pattern as a genuinely abandoned one. This isn't a
one-off failure but an underlying limitation of the kind of signals
being used.

Finally, "active" in this project means having received a recent push,
not being popular or well maintained. A repository with one trivial
commit last week counts the same as one under heavy development.

## 7. The web application

The application, called OSS Pulse, lets anyone enter the URL of a public
GitHub repository and get a live prediction. The backend, written in
Flask, receives the URL, queries the GitHub API to compute the same
first-month features used during training, and returns the probability
calculated by the model. The frontend, plain HTML, CSS, and JavaScript
with no framework, shows the result alongside the features used and a
few visualizations of the training dataset, to give the number some
context.

The diagram below shows what happens, layer by layer, from the moment a
URL is submitted to the moment a prediction comes back:

```mermaid
sequenceDiagram
    participant U as Browser (frontend)
    participant F as Flask backend
    participant G as GitHub API
    participant M as Gradient Boosting model

    U->>F: POST /predict {repo_url}
    F->>G: GET /repos/{owner}/{repo}
    G-->>F: metadata (stars, forks, license...)
    F->>G: commits, contributors (first 30 days)
    F->>G: GraphQL query — issue counts
    G-->>F: raw feature data
    F->>F: build feature vector (same as training)
    F->>M: predict_proba(features)
    M-->>F: survival probability
    F-->>U: JSON {prediction, probability, features_used}
```

The tool is most useful for recent repositories, weeks or a few months
old, where the question of whether they'll survive doesn't have an
observable answer yet. For repositories with years of history, checking
their recent activity directly is enough — no prediction is needed.

### Try it yourself

A few repositories worth testing, chosen to cover different profiles.
GitHub activity changes over time, so treat these as starting points
rather than guaranteed outcomes — the app itself is the ground truth at
the moment you run it:

- **Large, clearly active projects** — `pallets/flask`,
  `psf/requests`, `pandas-dev/pandas`. Expect a high survival
  probability: many contributors, licensed, high issue-close ratio.
- **A brand-new repository** (a few weeks old, created by you or a
  classmate) — this is the actual intended use case: a project too
  young to have an observable 6-month track record yet.
- **A solo academic or personal project** — try the author's own
  bachelor's thesis repo, `rsomavi/minidbms`, or one of your own. These
  tend to score as "likely abandoned" even when actively worked on,
  since they don't show the collaborative signals (multiple
  contributors, public issue tracking) the model learned from — a
  limitation discussed in Section 6.
- **An archived or explicitly discontinued project**, if you know one —
  useful to sanity-check the low end of the probability scale.

Pasting your own group's or classmates' repositories is a good way to
see the prediction, and its reasoning, applied to something you
actually recognize.

## 8. Conclusion

This project shows that it's possible to predict, with moderate but real
accuracy, whether a GitHub repository will stay active based on signals
from its first month of life. The most informative variables turned out
to be the issue close ratio and early contributor count, while others
that seemed relevant, like the wiki flag, contributed almost nothing to the model.

Beyond the numerical result, the process surfaced several lessons about
validating an end-to-end data pipeline: a sampling bias went unnoticed
until an impossible result forced a review of the collection design; a
pagination issue in the API quietly distorted the issue counts until
they were checked against a known ground truth; and a variable with
suspiciously high importance turned out to be tied to the problem's own
definition rather than carrying real signal. None of these three issues
were visible from the final results alone — all three came to light by
questioning numbers that, on the face of it, looked either too good or
too strange to be true.
