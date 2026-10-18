---
permalink: /
title: "About me"
excerpt: "Building agentic AI for finance"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% include agent-hero.html %}

<h2 class="agent-section-title">About</h2>

I am a curious and passionate scientist at heart. Trained as a statistician, I found my way through machine learning and statistical learning to build better models and solve impactful problems with data. Today I work on AI agents that plan, use tools and reason over financial data, so businesses (especially small ones) can forecast cash, manage risk and get better financing.

Here you can find my background, research, publications, and writing on AI, finance and life.

<h2 class="agent-section-title">Latest on AI</h2>

<ul>
{% assign shown = 0 %}
{% for post in site.posts %}{% if shown < 5 %}{% if post.type contains 'AI' or post.type == 'Data Science' %}{% assign shown = shown | plus: 1 %}
  <li><a href="{{ base_path }}{{ post.url }}">{{ post.title }}</a> <small style="color:#8b93a7">· {{ post.date | date: "%b %Y" }}</small></li>
{% endif %}{% endif %}{% endfor %}
</ul>
