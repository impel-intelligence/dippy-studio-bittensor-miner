# Dippy Bittensor Subnet FAQ

## General Overview

### What is the Dippy Bittensor Subnet?
The Dippy Studio Subnet (Subnet 11) is a decentralized network for distributed AI image inference tasks. 
Miners provide computational resources to process requests from validators.

### How does the subnet work?
1. Validators create inference jobs
2. An orchestrator routes each job to specific miners based on stake and performance
3. Miners process the requests and return results
4. Miners are scored based on their performance (currently speed for inference) and general completion
5. Emissions are distributed based on relative performance compared to other miners

### What types of jobs are there?
- **Inference jobs**: Currently active - miners process inference requests using the base model provided
- **Edit jobs**: Optional FLUX.1-Kontext editing

### How is miner selection determined?
The orchestrator uses a routing algorithm that considers:
- Miner stake (higher stake = higher priority)
- Miner performance scores
- Availability and capacity

### What model is being used?
All miners currently use the same base model for inference. In the future, lora inference will be introduced, allowing for more differentiation between miners.

## Mining and Requests

### How many requests should I expect to receive?
The current miner seleciton process is seen here: https://github.com/impel-intelligence/dippy-studio-bittensor/blob/a51cdf84c8d56d9143a189e2faf8354609845d18/orchestrator/clients/miner_metagraph.py#L226

Note that currently, this means miners with larger stake will likely see more requests received.
There is also a maximum effective alpha value currently set to 6000 alpha, which is subject to readjustment based on miner activity and capacity. As this is an ongoing measurement based on live data, expect multiple changes to this value.

## Emissions and Scoring

### Can I increase my emissions if more requests are routed?
Increasing the number of requests does not directly raise emissions. 
The emissions curve is based on relative scores - miners get better emissions by performing better compared to other miners, not by handling more requests.

### What determines the total score of each miner?
The current setup is based on completed requests. 
Score is absolute, while win rate is relative and calculated only against other miners that have an actual score (excluding miners with empty scores).

## Performance Evaluation

### How are inference jobs evaluated - by quality or speed?
Currently, inference jobs are measured by speed only. 
Quality is not applicable as all miners are utilizing the same base model. 
As lora inference is introduced, this will diverge over time and miners can reach the top by efficiently allocating lora inference resources.

### Is there a limit to inference processing time?
The maximum processing time depends on the request parameters. 
Currently, there's a parameter that refers to the number of steps, set at 50, but this will be variable in the future. 
It can scale up to 7 days theoretically, but this parameter is likely to be modified over time (linear for now).

### Are there ways to further optimize inference speed?
There are technical ways to optimize (overclocking, adding code optimizations, etc.), but that is up to the miner to implement.

## Technical Details

### Are all jobs sent to miners at the same time?
No, each job is created and sent to individual miners based on the routing algorithm.

### Where can I find the score and win rate calculation code?
The score to win rate conversion can be found in the validator code. 
Win rates and total wins are created there, while scores are fetched directly from the orchestrator.

### Is there a dashboard where miners can see scores of all miners?
You can run the validator code directly to fetch scores and the converted weights. 

## Troubleshooting

### Why haven't I received any requests?
If you have low stake, this is likely expected. Miners with larger amounts of stake are coming online and being prioritized for request distribution.


### I see "401 Unauthorized" in my logs. Should I be concerned?
This is someone trying to call your endpoint without the correct credentials. 
If you've ever operated an HTTP server before, this is a common occurrence and can be safely ignored.
