# Exercises

These exercises cover the data collection methods, study design concepts, and learning paradigms introduced in Chapter 1. Problems emphasize identifying study types, recognizing sources of bias, and distinguishing between statistical and machine learning approaches.

---

## Exercise 1: Population vs. Sample

For each scenario below, identify the **population** and the **sample**.

**(a)** A polling firm calls 1,200 randomly selected registered voters in a state to estimate the proportion who support a ballot measure.

**(b)** A hospital reviews the medical records of 500 patients admitted in 2024 to study post-surgical complications.

**(c)** An online retailer analyzes click data from 10,000 randomly selected user sessions to estimate the overall conversion rate.

### Solution

**(a)** Population: all registered voters in the state. Sample: the 1,200 voters who were called.

**(b)** Population: all patients admitted to the hospital (or, more precisely, all patients who could have been admitted). Sample: the 500 patients whose records were reviewed.

**(c)** Population: all user sessions on the retailer's website. Sample: the 10,000 selected sessions.

---

## Exercise 2: Parameter vs. Statistic

Classify each quantity as a **parameter** or a **statistic**.

**(a)** The average height of all adults in a country.

**(b)** The median income computed from a survey of 2,000 households.

**(c)** The proportion of defective items in an entire factory's production run.

**(d)** The standard deviation of test scores for 50 sampled students.

### Solution

**(a)** Parameter (a fixed quantity describing the entire population).

**(b)** Statistic (computed from a sample of 2,000 households).

**(c)** Parameter (describes the entire production run).

**(d)** Statistic (computed from a sample of 50 students).

---

## Exercise 3: Identifying Study Types

Classify each study as **observational** (cross-sectional, cohort, or case-control) or **experimental**.

**(a)** Researchers track 5,000 smokers and 5,000 non-smokers over 20 years to compare lung cancer rates.

**(b)** A pharmaceutical company randomly assigns 300 patients to receive either a new drug or a placebo and measures symptom improvement after 8 weeks.

**(c)** A public health team surveys 1,000 adults about their diet and exercise habits at a single point in time.

**(d)** Researchers identify 200 patients with heart disease and 200 without, then look back at their cholesterol history.

### Solution

**(a)** Observational -- cohort study (groups defined by exposure, followed forward in time).

**(b)** Experimental -- randomized controlled trial (random assignment to treatment groups).

**(c)** Observational -- cross-sectional study (data collected at a single time point).

**(d)** Observational -- case-control study (groups defined by outcome, looking backward at exposure).

---

## Exercise 4: Confounding Variables

A newspaper reports that cities with more ice cream trucks have higher crime rates, concluding that ice cream trucks cause crime.

**(a)** Identify a plausible **confounding variable** that explains the association.

**(b)** Explain why this study cannot establish a causal relationship between ice cream trucks and crime.

**(c)** Describe a study design that could better isolate the relationship, and explain why it may not be feasible.

### Solution

**(a)** Temperature (or population density) is a plausible confounder. Hot weather increases both ice cream consumption (more trucks) and outdoor activity (more opportunities for crime).

**(b)** This is an observational study. The observed association may be entirely due to the confounding variable. Without controlling for temperature and other confounders, the study cannot distinguish a causal effect from a spurious correlation.

**(c)** A randomized experiment would randomly assign ice cream trucks to cities and measure crime rates. This design is impractical and unethical because cities cannot be arbitrarily assigned numbers of ice cream trucks, and many other factors affect crime.

---

## Exercise 5: Randomization and Blinding

A clinical trial tests whether a new pain reliever works better than an existing one.

**(a)** Explain the purpose of **random assignment** in this trial.

**(b)** Define **single-blind**, **double-blind**, and **triple-blind** designs. Which would you recommend for this trial and why?

**(c)** If the new drug is a pill and the existing drug is an injection, what problem arises for blinding? How might researchers address it?

### Solution

**(a)** Random assignment distributes both known and unknown confounding variables (age, health status, pain tolerance) approximately equally across treatment groups. This ensures that observed differences in outcomes can be attributed to the treatment rather than to pre-existing differences.

**(b)**

- **Single-blind**: the patient does not know which treatment they receive, but the researcher does.
- **Double-blind**: neither the patient nor the researcher administering the treatment knows the assignment.
- **Triple-blind**: the patient, the administering researcher, and the data analyst are all blinded.

A double-blind design is recommended to prevent both patient expectation (placebo effect) and researcher bias from influencing the measured outcomes.

**(c)** Different delivery methods (pill vs. injection) make blinding difficult because the patient can tell which treatment they are receiving. Researchers can address this with a **double-dummy** design: every patient receives both a pill and an injection, one of which is a placebo (e.g., the new-drug group gets the real pill and a saline injection).

---

## Exercise 6: Sampling Methods

A university wants to survey student satisfaction. Identify the sampling method in each scenario.

**(a)** The registrar generates a list of all 20,000 students and uses a random number generator to select 500.

**(b)** The university divides students into freshmen, sophomores, juniors, and seniors, then randomly selects 125 from each class.

**(c)** The university randomly selects 10 dormitories and surveys every resident in those dormitories.

**(d)** A researcher stands outside the library and surveys the first 200 students who walk by.

### Solution

**(a)** Simple random sampling.

**(b)** Stratified random sampling (strata defined by class year).

**(c)** Cluster sampling (clusters are dormitories).

**(d)** Convenience sampling (not a probability-based method).

---

## Exercise 7: Identifying Bias

For each scenario, identify the **type of bias** (sampling bias, nonresponse bias, response bias, or survivorship bias).

**(a)** A magazine mails a political opinion survey to its subscribers. Only 15% respond, and respondents tend to hold stronger opinions than non-respondents.

**(b)** A mutual fund company advertises the strong performance of its current funds, without mentioning the funds that were closed due to poor performance.

**(c)** A telephone survey conducted during weekday business hours systematically misses working adults.

**(d)** In a face-to-face interview, respondents under-report their alcohol consumption.

### Solution

**(a)** Nonresponse bias -- the 85% who did not respond likely differ systematically from those who did.

**(b)** Survivorship bias -- only surviving (successful) funds are included, overstating average performance.

**(c)** Sampling bias (undercoverage) -- the sampling frame excludes people who are at work during calling hours.

**(d)** Response bias -- respondents give inaccurate answers due to social desirability.

---

## Exercise 8: Classical vs. Modern Approach

For each scenario, state whether the **classical** (design-first) or **modern** (data-first) approach is more appropriate, and justify your answer.

**(a)** A pharmaceutical company wants to determine whether a new vaccine reduces infection rates.

**(b)** An e-commerce company wants to predict which products a customer is likely to purchase next.

**(c)** A government agency wants to estimate the unemployment rate with a known margin of error.

### Solution

**(a)** Classical -- establishing a causal effect requires a randomized controlled trial with a designed protocol (random assignment, blinding, control group).

**(b)** Modern -- the company has existing transaction and browsing data. A prediction-focused algorithm (collaborative filtering, recommendation systems) can leverage this data without a new designed study.

**(c)** Classical -- precise estimation with a known margin of error requires a probability-based sample survey with a pre-specified sample size calculation.

---

## Exercise 9: Supervised, Unsupervised, or Reinforcement Learning

Classify each task into one of the three learning paradigms.

**(a)** Grouping customers into segments based on purchasing behavior, without predefined categories.

**(b)** Training a model to classify emails as spam or not spam using a labeled dataset.

**(c)** Teaching a robot to navigate a maze by rewarding it for reaching the exit and penalizing collisions.

**(d)** Predicting tomorrow's stock closing price given historical price and volume data.

**(e)** Detecting anomalous transactions in credit card data without labeled examples of fraud.

### Solution

**(a)** Unsupervised learning -- clustering without predefined labels.

**(b)** Supervised learning -- classification using labeled training data.

**(c)** Reinforcement learning -- an agent learns through trial-and-error interaction with an environment via rewards and penalties.

**(d)** Supervised learning -- regression with a continuous target variable.

**(e)** Unsupervised learning -- anomaly detection without labeled examples.

---

## Exercise 10: Prediction vs. Inference

For each research question, state whether the primary goal is **prediction** or **inference**, and explain the implications for method choice.

**(a)** Does a college degree cause higher lifetime earnings, controlling for ability and family background?

**(b)** Which customers are most likely to churn in the next 30 days?

**(c)** What is the effect of class size on student test scores?

### Solution

**(a)** Inference -- the goal is to estimate and test a causal effect. This requires controlling for confounders (ability, family background) using methods such as regression with controls, instrumental variables, or natural experiments. Model interpretability and valid standard errors are essential.

**(b)** Prediction -- the goal is to identify at-risk customers as accurately as possible. Flexible models (gradient-boosted trees, neural networks) that maximize predictive accuracy are appropriate, even if individual coefficients are not interpretable.

**(c)** Inference -- the goal is to estimate the causal effect of class size on outcomes. This requires careful attention to endogeneity (schools with more resources may have both smaller classes and better outcomes), suggesting methods such as randomized experiments or quasi-experimental designs.
