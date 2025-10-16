## Quick Start

python q_generator_shor_pro.py \
  --level L6 \
  --outdir circuits \
  --emit-json

python q_runner_shor_pro.py ./shor_out.qasm \
  --shots 2048 \
  --txt-out ../results/shor_out.counts.txt

python q_verify_shor_pro.py circuits/meta_private.json results --report --diagnose