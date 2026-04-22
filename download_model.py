from transformers import CLIPProcessor, CLIPModel

model_id  = "openai/clip-vit-base-patch32"

model     = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

model.save_pretrained(save_directory=r"Clip Model")
processor.save_pretrained(save_directory=r"Clip Model")
