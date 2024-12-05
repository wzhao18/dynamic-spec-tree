#!/bin/bash

run_command() {
  echo $@
  $@
}

# fix scenario to be T=0.6 P=0.9 CNN
run_command python3 testbed.py --T 0.6 --P 0.9 --M 512 --growmap ../L4_growmaps/7x128-64-384-64-stochastic.pt --Mode greedy

# tune different draft_T
run_command python3 testbed_dynamic.py --T 0.6 --P 0.9 --M 512 --draft_T 0.3 --tree_size 64
run_command python3 testbed_dynamic.py --T 0.6 --P 0.9 --M 512 --draft_T 0.4 --tree_size 64
run_command python3 testbed_dynamic.py --T 0.6 --P 0.9 --M 512 --draft_T 0.5 --tree_size 64
run_command python3 testbed_dynamic.py --T 0.6 --P 0.9 --M 512 --draft_T 0.6 --tree_size 64
run_command python3 testbed_dynamic.py --T 0.6 --P 0.9 --M 512 --draft_T 0.7 --tree_size 64
run_command python3 testbed_dynamic.py --T 0.6 --P 0.9 --M 512 --draft_T 0.8 --tree_size 64
run_command python3 testbed_dynamic.py --T 0.6 --P 0.9 --M 512 --draft_T 0.9 --tree_size 64
  
# tune different scaling
run_command python3 testbed_dynamic.py --T 0.6 --P 0.9 --M 512 --draft_T 0.6 --tree_size 64 --scaling_factor 0.8
run_command python3 testbed_dynamic.py --T 0.6 --P 0.9 --M 512 --draft_T 0.6 --tree_size 64 --scaling_factor 0.9
run_command python3 testbed_dynamic.py --T 0.6 --P 0.9 --M 512 --draft_T 0.6 --tree_size 64 --scaling_factor 1.1
run_command python3 testbed_dynamic.py --T 0.6 --P 0.9 --M 512 --draft_T 0.6 --tree_size 64 --scaling_factor 1.2

# experiment with different P
run_command python3 testbed.py --T 0.6 --P 0.7 --M 512 --growmap ../L4_growmaps/7x128-64-384-64-stochastic.pt --Mode greedy
run_command python3 testbed.py --T 0.6 --P 0.8 --M 512 --growmap ../L4_growmaps/7x128-64-384-64-stochastic.pt --Mode greedy
run_command python3 testbed.py --T 0.6 --P 0.9 --M 512 --growmap ../L4_growmaps/7x128-64-384-64-stochastic.pt --Mode greedy
run_command python3 testbed_dynamic.py --T 0.6 --P 0.7 --M 512 --draft_T 0.6 --tree_size 64
run_command python3 testbed_dynamic.py --T 0.6 --P 0.8 --M 512 --draft_T 0.6 --tree_size 64
run_command python3 testbed_dynamic.py --T 0.6 --P 0.9 --M 512 --draft_T 0.6 --tree_size 64

# experiment with offloading
run_command python3 testbed.py --T 0.6 --P 0.9 --M 512 --growmap ../L4_growmaps/7x128-64-384-64-stochastic.pt --Mode greedy --offloading
run_command python3 testbed_dynamic.py --T 0.6 --P 0.9 --M 512 --draft_T 0.6 --tree_size 64 --offloading

# experiment with different tree_size
run_command python3 testbed.py --T 0.6 --P 0.9 --M 512 --growmap ../L4_growmaps/7x16-384-64-stochastic.pt --Mode greedy
run_command python3 testbed.py --T 0.6 --P 0.9 --M 512 --growmap ../L4_growmaps/7x128-384-32-stochastic.pt --Mode greedy
run_command python3 testbed.py --T 0.6 --P 0.9 --M 512 --growmap ../L4_growmaps/7x128-64-384-64-stochastic.pt --Mode greedy
run_command python3 testbed.py --T 0.6 --P 0.9 --M 512 --growmap ../L4_growmaps/7x120-384-64-stochastic.pt --Mode greedy

run_command python3 testbed_dynamic.py --T 0.6 --P 0.9 --M 512 --draft_T 0.6 --tree_size 16
run_command python3 testbed_dynamic.py --T 0.6 --P 0.9 --M 512 --draft_T 0.6 --tree_size 32
run_command python3 testbed_dynamic.py --T 0.6 --P 0.9 --M 512 --draft_T 0.6 --tree_size 64
run_command python3 testbed_dynamic.py --T 0.6 --P 0.9 --M 512 --draft_T 0.6 --tree_size 128