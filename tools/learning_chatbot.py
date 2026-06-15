import os
import json
import shutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

SAVE_DIR = "trained_qwen_fast_style"
CHAT_LOG = "chat_log.jsonl"
STYLE_FILE = "user_style.json"

MAX_INPUT_TOKENS = 384
MAX_NEW_TOKENS = 70
MAX_STYLE_EXAMPLES = 10

# Higher = learns faster but can get weird faster.
LEARNING_RATE = 5e-4

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Qwen...")
print("Model:", MODEL_NAME)
print("Device:", device)
print()

if os.path.exists(SAVE_DIR):
    print("Loading trained version...")
    tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        SAVE_DIR, torch_dtype=torch.float32, trust_remote_code=True
    )
else:
    print("Downloading base Qwen...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True
    )

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.to(device)

# Freeze everything.
for p in model.parameters():
    p.requires_grad = False

# Train ONLY output head. This is much faster and changes output style quickly.
for p in model.lm_head.parameters():
    p.requires_grad = True

trainable_params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)

history = []


def load_json(path, default):
    if not os.path.exists(path):
        return default

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_jsonl(path, item):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_style():
    return load_json(STYLE_FILE, [])


def save_style(style):
    save_json(STYLE_FILE, style[-MAX_STYLE_EXAMPLES:])


def add_style(user_text):
    style = load_style()

    if len(user_text.strip()) < 2:
        return

    style.append(user_text.strip())
    save_style(style)


def style_text():
    style = load_style()

    if not style:
        return "No style examples yet."

    out = ""
    for s in style[-MAX_STYLE_EXAMPLES:]:
        out += "- " + s + "\n"

    return out.strip()


def make_messages(user_text):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a chatbot that copies the user's writing style.\n"
                "Match their tone, punctuation, capitalization, chaos level, sentence length, and wording.\n"
                "Do not sound polished unless the user sounds polished.\n"
                "Answer the actual message.\n\n"
                "User style examples:\n" + style_text()
            ),
        }
    ]

    for item in history[-6:]:
        messages.append(item)

    messages.append({"role": "user", "content": user_text})

    return messages


def generate_reply(user_text):
    model.eval()

    prompt = tokenizer.apply_chat_template(
        make_messages(user_text), tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        padding=True,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=1.05,
            top_p=0.95,
            repetition_penalty=1.02,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    if not reply:
        reply = "idk"

    return reply


def train_on_user_style(user_text):
    model.train()

    # This is the trick:
    # Instead of training on assistant replies, train Qwen to produce YOUR text.
    # That makes the model move toward your input style faster.
    messages = [
        {"role": "system", "content": "Write in the user's style."},
        {"role": "user", "content": "Say something in my style."},
        {"role": "assistant", "content": user_text},
    ]

    training_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    inputs = tokenizer(
        training_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        padding=True,
    ).to(device)

    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100

    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        labels=labels,
    )

    loss = outputs.loss

    if not torch.isfinite(loss):
        print("Bad loss. Skipped training.")
        return None

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

    optimizer.step()

    return float(loss.detach().cpu())


def save_model():
    os.makedirs(SAVE_DIR, exist_ok=True)
    tokenizer.save_pretrained(SAVE_DIR)
    model.save_pretrained(SAVE_DIR)


def reset():
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)

    if os.path.exists(STYLE_FILE):
        os.remove(STYLE_FILE)

    print("Deleted trained model and style file.")
    print("Restart script to reload clean Qwen.")
    print()


def main():
    print()
    print("FAST Qwen Style Trainer")
    print("-----------------------")
    print("Commands:")
    print("/exit   save and quit")
    print("/save   save now")
    print("/reset  delete trained model/style")
    print("/style  show style examples")
    print()
    print("This trains every message directly on YOUR text style.")
    print()

    count = 0

    while True:
        user_text = input("You: ").strip()

        if not user_text:
            continue

        if user_text == "/exit":
            print("Saving...")
            save_model()
            print("Saved. Goodbye.")
            break

        if user_text == "/save":
            print("Saving...")
            save_model()
            print("Saved.")
            print()
            continue

        if user_text == "/reset":
            reset()
            continue

        if user_text == "/style":
            print()
            for i, s in enumerate(load_style(), 1):
                print(str(i) + ". " + s)
            print()
            continue

        try:
            add_style(user_text)

            bot_text = generate_reply(user_text)

            print()
            print("Bot:", bot_text)
            print()

            # Train multiple tiny steps on the same user message.
            # Increase this if you want stronger/faster style learning.
            losses = []
            for _ in range(3):
                loss = train_on_user_style(user_text)
                if loss is not None:
                    losses.append(loss)

            history.append({"role": "user", "content": user_text})

            history.append({"role": "assistant", "content": bot_text})

            append_jsonl(
                CHAT_LOG, {"user": user_text, "bot": bot_text, "losses": losses}
            )

            if losses:
                print("Trained. Loss:", round(losses[-1], 4))
            else:
                print("Training skipped.")

            print()

            count += 1

            # Saving is slow, so do it less often.
            if count % 25 == 0:
                print("Auto-saving...")
                save_model()
                print("Saved.")
                print()

        except RuntimeError as e:
            print()
            print("Runtime error:", e)
            print("If this says CUDA device-side assert, restart Python.")
            print()

        except KeyboardInterrupt:
            print()
            print("Saving...")
            save_model()
            print("Saved. Goodbye.")
            break

        except Exception as e:
            print()
            print("Error:", e)
            print()


if __name__ == "__main__":
    main()
