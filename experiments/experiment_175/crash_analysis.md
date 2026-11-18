CLASSIFICATION: INFRA

RunPod serverless deployment failed with: `endpoint template names must be unique. check that you do not already have an endpoint with a similar template.`

The training code never executed on the GPU. This is a RunPod API-side naming collision — likely a stale endpoint from the exp174 INFRA crash wasn't cleaned up, and exp175 tried to create an endpoint with the same or similar template name.

The train.py code itself is valid and has no bugs. Retrying (possibly after clearing stale RunPod endpoints) should work.
