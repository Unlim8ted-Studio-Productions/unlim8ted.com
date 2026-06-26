# pt_to_onnx_gui.py
#
# Simple GUI for converting a trained .pt checkpoint to ONNX.
#
# Requirements:
# pip install torch onnx tkinter
#
# Usage:
# python pt_to_onnx_gui.py
#
# Assumes the .pt checkpoint contains:
# {
#     "model_state_dict": ...
# }
#
# Edit build_model() to match your architecture.

import json
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import torch
import torch.nn as nn

# ============================================================
# CHANGE THESE TO MATCH YOUR MODEL
# ============================================================

PROMPT_SIZE = 128
HIDDEN_SIZE = 192
EMBED_SIZE = 128
DROPOUT = 0.35

MAX_OUTPUT_CHUNKS = 24

PAD_ID = 0
BOS_ID = 1


# ============================================================
# MODEL DEFINITION
# Replace if architecture differs
# ============================================================


class ManualGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))

        self.bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(3 * hidden_size))

        self.hidden_size = hidden_size
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / (self.hidden_size**0.5)

        for w in self.parameters():
            nn.init.uniform_(w, -stdv, stdv)

    def forward(self, x, h):
        gi = torch.matmul(x, self.weight_ih.t()) + self.bias_ih
        gh = torch.matmul(h, self.weight_hh.t()) + self.bias_hh

        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)

        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)

        n = torch.tanh(i_n + r * h_n)

        return n + z * (h - n)


class ChunkAnswerModel(nn.Module):
    def __init__(self, input_size, output_vocab_size):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, PROMPT_SIZE),
            nn.LayerNorm(PROMPT_SIZE),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(PROMPT_SIZE, PROMPT_SIZE),
            nn.LayerNorm(PROMPT_SIZE),
            nn.GELU(),
            nn.Dropout(0.0),
        )

        self.embedding = nn.Embedding(output_vocab_size, EMBED_SIZE)

        self.decoder_cell = ManualGRUCell(PROMPT_SIZE + EMBED_SIZE, HIDDEN_SIZE)

        self.output = nn.Sequential(
            nn.LayerNorm(PROMPT_SIZE + HIDDEN_SIZE),
            nn.Linear(PROMPT_SIZE + HIDDEN_SIZE, output_vocab_size),
        )

    def encode(self, x):
        return self.encoder(x)

    def decoder_step(
        self,
        prev_token,
        prompt_context,
        write_hidden,
    ):
        emb = self.embedding(prev_token)

        write_hidden = self.decoder_cell(
            torch.cat([emb, prompt_context], dim=-1),
            write_hidden,
        )

        logits = self.output(
            torch.cat(
                [prompt_context, write_hidden],
                dim=-1,
            )
        )

        return logits, write_hidden

    def forward(self, x, max_len=MAX_OUTPUT_CHUNKS + 1):
        batch_size = x.size(0)

        prompt_context = self.encode(x)

        write_hidden = torch.zeros(
            batch_size,
            HIDDEN_SIZE,
            device=x.device,
        )

        prev_token = torch.full(
            (batch_size,),
            BOS_ID,
            dtype=torch.long,
            device=x.device,
        )

        logits_steps = []

        for _ in range(max_len):
            logits, write_hidden = self.decoder_step(
                prev_token,
                prompt_context,
                write_hidden,
            )

            logits_steps.append(logits.unsqueeze(1))

            prev_token = torch.argmax(logits, dim=-1)

        return torch.cat(
            logits_steps,
            dim=1,
        )


# ============================================================
# BUILD MODEL
# ============================================================


def build_model(checkpoint, input_vocab_path, output_chunks_path):

    with open(input_vocab_path, "r", encoding="utf-8") as f:
        input_vocab = json.load(f)

    with open(output_chunks_path, "r", encoding="utf-8") as f:
        output_chunks = json.load(f)

    model = ChunkAnswerModel(
        input_size=len(input_vocab),
        output_vocab_size=len(output_chunks),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False,
    )

    model.eval()

    return model, len(input_vocab)


# ============================================================
# CONVERSION
# ============================================================


def convert_checkpoint():

    try:

        pt_path = filedialog.askopenfilename(
            title="Select .pt checkpoint", filetypes=[("PyTorch", "*.pt")]
        )

        if not pt_path:
            return

        input_vocab_path = filedialog.askopenfilename(
            title="Select input_vocab.json", filetypes=[("JSON", "*.json")]
        )

        if not input_vocab_path:
            return

        output_chunks_path = filedialog.askopenfilename(
            title="Select output_chunks.json", filetypes=[("JSON", "*.json")]
        )

        if not output_chunks_path:
            return

        save_path = filedialog.asksaveasfilename(
            title="Save ONNX as",
            defaultextension=".onnx",
            filetypes=[("ONNX", "*.onnx")],
        )

        if not save_path:
            return

        status_var.set("Loading checkpoint...")
        root.update()

        checkpoint = torch.load(
            pt_path,
            map_location="cpu",
        )

        model, input_size = build_model(
            checkpoint,
            input_vocab_path,
            output_chunks_path,
        )

        dummy = torch.zeros(
            1,
            input_size,
            dtype=torch.float32,
        )

        status_var.set("Exporting...")
        root.update()

        torch.onnx.export(
            model,
            dummy,
            save_path,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch"},
                "logits": {0: "batch"},
            },
            opset_version=17,
        )

        status_var.set("Done")

        messagebox.showinfo("Success", f"Saved:\n{save_path}")

    except Exception:

        status_var.set("Failed")

        traceback.print_exc()

        messagebox.showerror("Error", traceback.format_exc())


# ============================================================
# GUI
# ============================================================

root = tk.Tk()
root.title("PT → ONNX Converter")
root.geometry("450x180")

frame = tk.Frame(root)
frame.pack(expand=True)

title = tk.Label(frame, text="PyTorch → ONNX Converter", font=("Arial", 16, "bold"))
title.pack(pady=10)

btn = tk.Button(
    frame,
    text="Select Checkpoint and Convert",
    command=convert_checkpoint,
    width=35,
    height=2,
)
btn.pack(pady=10)

status_var = tk.StringVar()
status_var.set("Ready")

status = tk.Label(
    frame,
    textvariable=status_var,
)
status.pack()

root.mainloop()
