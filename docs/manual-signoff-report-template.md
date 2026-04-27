# Painel Amanaje Manual Signoff Report Template

## Run Summary

- Tester:
- Date:
- Branch / commit:
- Environment:
- Pass type: `Fresh stack` / `Stateful recovery`
- Final result: `PASS` / `FAIL`

## Stack Health

- API URL:
- MLflow URL:
- Airflow URL:
- pgAdmin URL:
- Postgres reachable: `Yes` / `No`
- Notes:

## Fixture Set Used

- Dataset fixtures:
- Model fixtures:
- Manifest checked: `Yes` / `No`
- Notes:

## Checklist Outcome

| Section | Status | Notes |
| --- | --- | --- |
| A. Stack, Shell, and Global Layout |  |  |
| B. Upload |  |  |
| C. Create |  |  |
| D. Editor |  |  |
| E. Feature Workspace |  |  |
| F. Training, Optimization, and ONNX |  |  |
| G. Production, Monitoring, Simulation, and MLflow |  |  |
| H. Registry |  |  |
| I. Direct JSON/API Surface Checks |  |  |
| J. Stateful Recovery, Restarts, and Stress-Like Checks |  |  |
| K. Final Sweep |  |  |

## Failure Log

Use one block per issue.

### Failure 1

- URL:
- Active tab / panel:
- Exact user action:
- Expected result:
- Actual result:
- Browser console error:
- Failing request:
- Response body:
- Reproduces on refresh: `Yes` / `No`
- Severity:
- Screenshot / evidence path:

### Failure 2

- URL:
- Active tab / panel:
- Exact user action:
- Expected result:
- Actual result:
- Browser console error:
- Failing request:
- Response body:
- Reproduces on refresh: `Yes` / `No`
- Severity:
- Screenshot / evidence path:

## Stateful Recovery Notes

- Existing datasets created:
- Existing models created:
- Existing studies created:
- Existing inference pairs created:
- Existing training/study runs reused:
- Existing production watch contexts reused:

## Signoff Notes

- Highest-risk area:
- Unexpected behavior that did not become a failure:
- Recommended next fix or follow-up:
