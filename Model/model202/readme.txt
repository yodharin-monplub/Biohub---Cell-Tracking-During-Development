MODEL202 - SETUP SCRIPT FOR RUNNING THE FROZEN 0.947 PIPELINE ON A RENTED GPU (not used further)

remote_setup.sh prepares a rented Linux GPU machine (Vast.ai) to run the frozen 0.947 notebook pipeline on the 4
visible test movies: a Python 3.12 venv (the public support-pack wheels are cp312), torch 2.5.1 + CUDA 12.4, and
the pipeline's own offline wheels from Data\public_checkpoints\support-pack\wheels. It expects the project payload
unpacked from /root/payload.tar.

Status: infrastructure only - no model was trained and nothing was scored here. The comparable local run of the
frozen pipeline on the 4 visible test movies is Model\model199 (laptop GPU).
