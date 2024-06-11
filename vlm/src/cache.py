from inference import load_model, get_grounding_output, load_image

checkpoint = "checkpoint_best_regular.pth"
config = "GroundingDINO_SwinT_OGC.py"
        
model = load_model(config, checkpoint)