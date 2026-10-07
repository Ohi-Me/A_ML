# Team Enigma (Rohit Kumar, Vishesh Shekhawat, Sriyansh, Veeky Kumar) - ML Challenge 2026, Business Entity Resolution
echo "PBS-provided CUDA_VISIBLE_DEVICES (before our override): $PBS_CUDA_ORIG"
env | grep -i -E "cuda|gpu|pbs_" | sort
nvidia-smi -L | head -30
cat /proc/self/cgroup
