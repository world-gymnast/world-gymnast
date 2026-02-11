<div align="center">

## World-Gymnast: Training Robots with Reinforcement Learning in a World Model

</div>

**World-Gymnast** fine-tunes vision-language-action policies by rolling out actions in a learned world model and scoring imagined trajectories with a vision-language model. It improves real-robot performance, supports training with distractors and novel language, and enables test-time training plus iterative world-model updates.

<div align="center">
  <img src="figs/main_figure.png" alt="World-Gymnast main figure" width="90%"/>
</div>

## Setup

See [SETUP.md](SETUP.md) for environment setup and dependencies.

## Running World-Gymnast

Example training script:
- `examples/run_openvla_oft_rl_worldgym.sh`

## Data

Training data is stored as JSON annotations plus PNG images.

Example JSON:
```json
{"instruction": "lift eggplant", "partial_credit_criteria": "the robot makes contact with the eggplant"}
```

## Project Website

https://world-gymnast.github.io/

## Paper

https://arxiv.org/abs/2602.02454

## Citation

```bibtex
@misc{sharma2026worldgymnasttrainingrobotsreinforcement,
      title={World-Gymnast: Training Robots with Reinforcement Learning in a World Model},
      author={Ansh Kumar Sharma and Yixiang Sun and Ninghao Lu and Yunzhe Zhang and Jiarao Liu and Sherry Yang},
      year={2026},
      eprint={2602.02454},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.02454},
}
```

## Acknowledgements
- [WorldGym](https://github.com/world-model-eval/world-model-eval)
- [SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL)
- [OpenVLA-OFT](https://github.com/moojink/openvla-oft)
