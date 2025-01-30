import os
import PIL.Image
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
import time
import re

# Specify the path to the model
model_path = "deepseek-ai/Janus-Pro-7B"
vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

def generate_images(prompt, num_images=1):
    conversation = [
        {"role": "User", "content": prompt},
        {"role": "Assistant", "content": ""}
    ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt=""
    )
    full_prompt = sft_format + vl_chat_processor.image_start_tag
    
    with torch.inference_mode():
        input_ids = torch.LongTensor(vl_chat_processor.tokenizer.encode(full_prompt)).cuda()
        tokens = input_ids.repeat(num_images * 2, 1)
        for i in range(num_images * 2):
            if i % 2 != 0:
                tokens[i, 1:-1] = vl_chat_processor.pad_id

        inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
        generated_tokens = torch.zeros((num_images, 576), dtype=torch.int).cuda()

        for i in range(576):
            outputs = vl_gpt.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=outputs.past_key_values if i != 0 else None
            )
            logits = vl_gpt.gen_head(outputs.last_hidden_state[:, -1, :])
            logits = logits[1::2, :] + 5 * (logits[0::2, :] - logits[1::2, :])
            
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            
            next_token = torch.cat([next_token.unsqueeze(1)] * 2, dim=1).view(-1)
            inputs_embeds = vl_gpt.prepare_gen_img_embeds(next_token).unsqueeze(1)

        images = vl_gpt.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[num_images, 8, 24, 24]
        )
        images = ((images.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1) + 1) / 2 * 255).clip(0, 255)

        os.makedirs('generated_samples', exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        short_prompt = re.sub(r'\W+', '_', prompt)[:50]

        for i in range(num_images):
            save_path = os.path.join('generated_samples', f"img_{timestamp}_{short_prompt}_{i}.jpg")
            PIL.Image.fromarray(images[i].astype(np.uint8)).save(save_path)

def main():
    print("Welcome to the Janus Image Generator!")
    while True:
        try:
            num_images = int(input("How many images to generate? (1-16): "))
            if 1 <= num_images <= 16:
                break
            print("Please enter a number between 1 and 16")
        except ValueError:
            print("Please enter a valid number")

    while True:
        prompt = input("\nEnter image description (or 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break
            
        print(f"\nGenerating {num_images} image(s) for: '{prompt}'")
        generate_images(prompt, num_images)
        print("Done! Check the 'generated_samples' folder")

if __name__ == "__main__":
    main()