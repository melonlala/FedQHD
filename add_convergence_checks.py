"""Add convergence checks to all training functions"""
import re

# Read the file
with open('experiments_runner.py', 'r') as f:
    content = f.read()

# Pattern to find training loops
# We'll add convergence check after reward_history.append() lines

# Find all places where we append to reward_history and add convergence check after
pattern = r'(        reward_history\.append\(np\.mean\(episode_rewards\)\)\n        success_history\.append\(np\.mean\(episode_successes\)\))'

replacement = r'''\1

        # Check for convergence
        if check_convergence(reward_history):
            print(f"\\nConverged at episode {episode + 1}")
            break'''

# Apply the replacement
new_content = re.sub(pattern, replacement, content)

# Write back
with open('experiments_runner.py', 'w') as f:
    f.write(new_content)

print("Added convergence checks to all training functions!")
print(f"Added {new_content.count('Check for convergence')} convergence checks")
