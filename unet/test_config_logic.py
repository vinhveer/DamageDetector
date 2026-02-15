import sys
import argparse
from unittest.mock import MagicMock

# Mocking sys.argv to simulate CLI usage
# Case 1: User provides only required args. Should load defaults from YAML.
sys.argv = [
    "train.py",
    "--train-images", "dummy_train",
    "--train-masks", "dummy_masks",
    "--val-images", "dummy_val",
    "--val-masks", "dummy_val_masks"
]

# Mock config dictionary simulating YAML content
model_config = {
    "training": {
        "learning_rate": 0.0001, # Different from CLI default (0.0005)
        "pos_weight": 2.0,       # Different from CLI default (5.0)
        "no_augment": True,      # Different from CLI default (False)
    },
    "dataloader": {
        "num_workers": 4         # Different from CLI default (8)
    },
    "model": {}
}

training_cfg = model_config.get("training", {})
dataloader_cfg = model_config.get("dataloader", {})

# The logic to test
def override_if_default(arg_val, arg_name, yaml_key, default_val, section=training_cfg):
    cli_flag = "--" + arg_name.replace("_", "-")
    is_passed = any(cli_flag in s for s in sys.argv)
    if not is_passed and yaml_key in section:
            return section[yaml_key]
    return arg_val

def test_logic():
    print("--- Test Case 1: No optional args passed in CLI ---")
    
    # 1. Learning Rate
    # CLI default is 0.0005. YAML has 0.0001. User didn't pass --learning-rate.
    # Expected: 0.0001 (YAML)
    lr_val = override_if_default(0.0005, "learning_rate", "learning_rate", 0.0005)
    print(f"Learning Rate: Expected 0.0001 | Got: {lr_val} | {'PASS' if lr_val == 0.0001 else 'FAIL'}")

    # 2. Pos Weight
    # CLI default "5.0". YAML has 2.0.
    pos_val = 5.0
    if "--pos-weight" not in sys.argv and "pos_weight" in training_cfg:
         pos_val = training_cfg["pos_weight"]
    print(f"Pos Weight: Expected 2.0 | Got: {pos_val} | {'PASS' if pos_val == 2.0 else 'FAIL'}")

    # 3. Augment
    # CLI default False. YAML True.
    no_aug = False
    if "no_augment" in training_cfg and training_cfg["no_augment"] and "--no-augment" not in sys.argv:
         no_aug = True
    print(f"No Augment: Expected True | Got: {no_aug} | {'PASS' if no_aug is True else 'FAIL'}")

    print("\n--- Test Case 2: User explicitly passes args ---")
    # Simulate user passing --learning-rate 0.002
    sys.argv.append("--learning-rate")
    sys.argv.append("0.002")
    
    # Expected: 0.002 (CLI) - because is_passed is True
    lr_val_2 = override_if_default(0.002, "learning_rate", "learning_rate", 0.0005)
    print(f"Learning Rate (Override): Expected 0.002 | Got: {lr_val_2} | {'PASS' if lr_val_2 == 0.002 else 'FAIL'}")

if __name__ == "__main__":
    test_logic()
