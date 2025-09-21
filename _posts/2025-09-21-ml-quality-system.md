# Summary: *How good are your ML best practices?*  
*(George Chouliaras et al., Booking.com Data Science)*  
[Read the full article](https://booking.ai/how-good-are-your-ml-best-practices-fd7722262437)

---

## Overview
The article proposes a framework to help organizations **evaluate, prioritize, and adopt ML best practices** by linking them to measurable *quality attributes*. Since not all practices are equally useful in every context, the framework provides a systematic way to select practices based on impact and constraints.

---

## Key Components

### 1. Quality Model for ML
- Defines “quality sub-characteristics” relevant for ML systems.  
- Examples: **accuracy, robustness, scalability, testability, understandability, discoverability, compliance**.

### 2. Best Practices Inventory
- Collected from literature + internal surveys.  
- Examples:  
  - Versioning data, model, and config  
  - Continuous monitoring of deployed models  
  - Automating workflows  
  - Removing redundant features  
  - Testing feature code  
  - Containerized environments  
  - Documentation  
  - Modular/reusable code

### 3. Mapping Practices to Quality
- Each practice is scored **0–4** against how much it contributes to each sub-characteristic.  
- Ratings were provided by ML engineers and scientists.

### 4. Prioritization / Optimization
- Organizations can’t adopt all practices.  
- Optimization (greedy or brute force) is used to **pick subsets of practices** that maximize quality coverage under resource constraints.

---

## Findings & Insights
- **Coverage statistics**:  
  - ~5 practices → ~40% coverage  
  - ~10 practices → ~70% coverage  
  - ~24 practices → ~96% coverage (diminishing returns after this)

- **Top impactful practices include**:  
  - Versioning for data, model, configs/scripts  
  - Continuous monitoring of model behaviour  
  - Automating ML workflows  
  - Removing redundant features  
  - Testing input-feature code  
  - Shadow deployment  
  - Constant performance measurement  
  - Documentation  
  - Modular & reusable code

- **Not all quality goals are equally covered**: weaker areas include *scalability*, *discoverability*, *standards compliance*.  
- “Optimal” depends on what your organization values most (e.g., robustness vs. compliance).

---

## Recommendations
- Don’t adopt practices blindly — **prioritize based on your quality goals**.  
- **Map practices to quality attributes** to reveal overlaps and gaps.  
- Adopt only the **most impactful practices** until diminishing returns set in.  
- Use structured optimization methods to decide where to start.  
- Revisit your practices regularly to adapt to changing priorities.

---
