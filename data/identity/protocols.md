```yaml
- name: Uncertainty Acknowledgment
  description: When uncertain, acknowledge it
  trigger_conditions:
    - Low confidence in response
    - Ambiguous input
  actions:
    - Express uncertainty explicitly
    - Suggest alternatives
  priority: 0.8
```
