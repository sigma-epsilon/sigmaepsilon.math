{{ fullname }}
{{ underline }}

.. automodule:: {{ fullname }}

{% block classes %}
{% if classes %}
.. rubric:: Classes

.. autosummary::
   :toctree:
   :template: custom-class-template.rst
{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block functions %}
{% if functions %}
.. rubric:: Functions

.. autosummary::
   :toctree:
{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
