"""Distillation: student model trained to match teacher logits."""
import torch
import torch.nn.functional as F


def distill_loss(student_logits, teacher_logits, labels, alpha=0.5, temperature=4.0):
    """Combined CE + KL loss."""
    ce = F.cross_entropy(student_logits, labels)
    soft_kl = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)
    return alpha * ce + (1 - alpha) * soft_kl


def train_step(student, teacher, batch, optimizer):
    student.train()
    teacher.train(False)
    inputs, labels = batch
    with torch.no_grad():
        teacher_out = teacher(inputs)
    student_out = student(inputs)
    loss = distill_loss(student_out, teacher_out, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()
