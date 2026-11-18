CLASSIFICATION: INFRA

RunPod endpoint deployment failed with: `endpoint template names must be unique. check that you do not already have an endpoint with a similar template.`

The training script never reached the GPU. This is a RunPod GraphQL API error during serverless endpoint creation — a stale/duplicate endpoint template exists from a previous run. The training code (train.py) was never executed, so there is no code bug. Retrying (possibly after cleaning up old RunPod endpoints) should resolve this.
