---
layout: archive
title: "AI"
permalink: /ai/
author_profile: true
---

{% include base_path %}

<div class="agent-term" style="margin-bottom:2em">
  <div class="agent-term__bar">
    <span class="agent-term__dots"><i></i><i></i><i></i></span>
    <span class="agent-term__title">focus.yaml</span>
  </div>
<pre class="agent-term__body"><span class="t-step">focus</span>:   agentic AI for finance
<span class="t-tool">agents</span>:  planning · tool use · memory · self-evaluation
<span class="t-tool">domain</span>:  cash forecasting · treasury · credit risk · lending
<span class="t-tool">craft</span>:   evaluation · ML quality · getting models to production</pre>
</div>

Notes and summaries on LLMs, agents and the ML systems behind them.

{% for post in site.posts %}{% if post.type contains 'AI' or post.type == 'Data Science' %}
  {% include archive-single.html %}
{% endif %}{% endfor %}
