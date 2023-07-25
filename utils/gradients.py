import torch
import numpy.ma as ma
from scipy import spatial
from utils.multitask_losses import compute_mtloss

def record_task_gradients(model, targeted, images_metrics, labels, keep_graph=False, return_outputs=False):
    gradients = {}
    criterion = torch.nn.BCEWithLogitsLoss()

    for (t, label) in labels.items():
        if t == "rep":
            continue

        images = images_metrics.clone().to(label.device)
        images.requires_grad = True
        outputs = model(images)

        loss, loss_dict, avg_losses = compute_mtloss({t: criterion}, outputs, labels, equally=True,
                                                     loss_dict={}, avg_losses=None)

        cost = -loss if targeted else loss

        grad = torch.autograd.grad(cost, images,
                                   retain_graph=keep_graph, create_graph=keep_graph)[0]

        gradients[t] = grad if keep_graph else grad.detach().cpu()

    return gradients if return_outputs is False else (gradients,outputs)


def record_metric_gradients(model,targeted, images, labels, self_gradients, self_labels, self_gradients_cov,
                            self_gradients_cosine, self_gradients_dot, self_gradients_magn, self_gradients_curve):
    gradients = record_task_gradients(model,targeted, images, labels)
    self_gradients.append(gradients)
    self_labels.append(labels)

    if len(list(gradients.keys())) > 1:
        with torch.no_grad():
            gradients_vals = list(gradients.values())
            task_1 = gradients_vals[0].flatten()
            task_2 = gradients_vals[1].flatten()
            summ = task_1 + task_2
            diff = task_1 - task_2
            norm_1 = task_1.norm()
            norm_2 = task_2.norm()

            cov = ma.cov(ma.masked_invalid(task_1.numpy()),
                         ma.masked_invalid(task_2.numpy())).tolist()

            cosine_similarity = torch.nn.CosineSimilarity()(gradients_vals[0], gradients_vals[1])
            similarity = cosine_similarity.detach().numpy().tolist()

            dot_product = torch.dot(task_1, task_2)
            cos_angle = dot_product / (norm_1 * norm_2)
            magnitude_similarity = 2 * norm_2 * norm_1 / (norm_1 ** 2 + norm_2 ** 2)
            curvature_measure = (1 - cos_angle ** 2) * diff.norm() ** 2 / summ.norm() ** 2
            # print(f"gradient - similarity {similarity} covariance {cov} dot {dot_product}")

            self_gradients_cov.append(cov)
            self_gradients_cosine.append(similarity)
            self_gradients_dot.append(dot_product)
            self_gradients_magn.append(magnitude_similarity)
            self_gradients_curve.append(curvature_measure)

    return self_gradients, self_labels, self_gradients_cov, self_gradients_cosine, self_gradients_dot, self_gradients_magn, self_gradients_curve