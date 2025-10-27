import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scripts.utils import *


SCALE = 11/8
FIGURE_DIR = os.path.join(FIGURE_CACHE, 'f2b')
os.makedirs(FIGURE_DIR, exist_ok=True)


def param_size_format(size):
    # to human readable format
    if size < 1e3:
        return f"{size}"
    elif size < 1e6:
        return f"{size/1e3:.0f}K"
    elif size < 1e9:
        return f"{size/1e6:.0f}M"
    else:
        return f"{size/1e9:.0f}B"



def main(args):
    OTHERS = ["DorsalNet", "blt_temporal"]
    GROUPS, means, stds, group_models = pickle_store.load(f"cache.tests.paper_plots.f1.model_scores.{args.clip_duration}")[args.region]
    meta, models = collect(args, types=['meta'], exclude_pixels=False, models=ALL_MODELS+OTHERS)
    fps = {name: meta[name]["fps"] for name in models}
    sizes = {name: meta[name]["params"] for name in models}

    all_groups = np.array(sum([[group] * len(mean) for group, mean in zip(GROUPS, means)], []))
    all_means = np.concatenate(means)
    all_stds = np.concatenate(stds)
    all_models = np.concatenate([group_models[group] for group in GROUPS])
    
    # sort by mean
    indices = np.argsort(all_means)
    groups = all_groups[indices]
    all_means = all_means[indices]
    all_stds = all_stds[indices]
    all_models = all_models[indices]

    # stats test
    from scipy.stats import mannwhitneyu
    baslines = all_means[np.where(groups == "Baselines")[0]]
    untrained = all_means[np.where(groups == "Untrained Models")[0]]
    image_models = all_means[np.where(groups == "Image Recognition")[0]]
    forward_pred = all_means[np.where(groups == "Forward Prediction")[0]]
    audio_video = all_means[np.where(groups == "Audio-Video")[0]]
    action_rec = all_means[np.where(groups == "Action Recognition")[0]]
    masked_auto = all_means[np.where(groups == "Masked Autoencoder")[0]]
    text_video = all_means[np.where(groups == "Text-Video")[0]]

    other_c = "yellow"
    cs = [colors.BASELINE_1, colors.BASELINE_2, *colors.MODELS] + [other_c]*len(np.where(groups == "Others")[0]) 

    pvals = []
    # Baselines vs Untrained Models
    pvals.append(mannwhitneyu(baslines, untrained).pvalue)
    # Untrained Models vs Image Recognition
    pvals.append(mannwhitneyu(untrained, image_models).pvalue)
    # Image Recognition vs Forward Prediction
    pvals.append(mannwhitneyu(image_models, forward_pred).pvalue)
    # Image Recognition vs Audio-Video
    pvals.append(mannwhitneyu(image_models, audio_video).pvalue)
    # Image Recognition vs Action Recognition
    pvals.append(mannwhitneyu(image_models, action_rec).pvalue)
    # Image Recognition vs Masked Autoencoder
    pvals.append(mannwhitneyu(image_models, masked_auto).pvalue)
    # Image Recognition vs Text-Video
    pvals.append(mannwhitneyu(image_models, text_video).pvalue)

    pvals = false_discovery_control(pvals)
    print(f"""
        Baselines vs Untrained Models: {pval_to_stars(pvals[0]), pvals[0]}
        Untrained Models vs Image Recognition: {pval_to_stars(pvals[1]), pvals[1]}
        Image Recognition vs Forward Prediction: {pval_to_stars(pvals[2]), pvals[2]}
        Image Recognition vs Audio-Video: {pval_to_stars(pvals[3]), pvals[3]}
        Image Recognition vs Action Recognition: {pval_to_stars(pvals[4]), pvals[4]}
        Image Recognition vs Masked Autoencoder: {pval_to_stars(pvals[5]), pvals[5]}
        Image Recognition vs Text-Video: {pval_to_stars(pvals[6]), pvals[6]}
    """)

    # for paper
    print(f"Best baseline: {baslines.max()}")
    print(f"Trained mean: {np.concatenate([image_models, forward_pred, audio_video, action_rec, masked_auto, text_video]).mean()}")
    print(f"Best model: {np.concatenate([image_models, forward_pred, audio_video, action_rec, masked_auto, text_video]).max()}")

    # # move baseline to the front
    # baseline_index = np.where(groups == "Baselines")[0]
    # non_baseline_index = np.where(groups != "Baselines")[0]
    # groups = np.concatenate([["Baselines"]*len(baseline_index), groups[non_baseline_index]])
    # all_means = np.concatenate([all_means[baseline_index], all_means[non_baseline_index]])
    # all_stds = np.concatenate([all_stds[baseline_index], all_stds[non_baseline_index]])

    xs = np.arange(len(all_means))

    # Figure: inset
    plt.subplots(figsize=(3.5*SCALE, 2))
    for i, (group, c) in enumerate(zip(GROUPS, cs)):
        group_indices = np.where(groups == group)[0]
        plt.bar(
            xs[group_indices], all_means[group_indices], yerr=all_stds[group_indices], label=group,
            linewidth=0, edgecolor="none", width=1, error_kw=dict(lw=0.5), color=c
        )

    plt.xticks([])
    plt.title("Individual models")
    sns.despine()
    plt.ylim(0, 0.6)

    plt.savefig(f"{FIGURE_DIR}/f1_score_bar-inset-{args.region}-{args.clip_duration}.svg", bbox_inches="tight", dpi=300)
    plt.close()

    # Figure: details
    plt.subplots(figsize=(17, 3))
    x_ticks = [None] * len(xs)
    for i, (group, c) in enumerate(zip(GROUPS, cs)):
        group_indices = np.where(groups == group)[0]
        plt.bar(
            xs[group_indices], all_means[group_indices], yerr=all_stds[group_indices], label=group,
            linewidth=0, edgecolor="none", width=1, error_kw=dict(lw=1), color=c
        )

        this_group = all_models[group_indices]

        # print score for each model
        for j, (m, mean, std) in enumerate(zip(this_group, all_means[group_indices], all_stds[group_indices])):
            print(f"{group} {j+1}: {m} {mean:.3f} ± {std:.3f}")

        # annotate the fps and formatted sizes on the top, rotate 90 degree
        fps_disp = [f"FPS:{fps[model]}" for model in this_group]
        size_disp = [param_size_format(sizes[model]) for model in this_group]
        text_disp = [f"{fps_}  {size_}" for fps_, size_ in zip(fps_disp, size_disp)]

        for j, (x, y, fps_, size_) in enumerate(zip(xs[group_indices], all_means[group_indices], fps_disp, size_disp)):
            plt.text(x, y+0.05, text_disp[j], ha="center", va="bottom", rotation=90, fontsize=10)

        for i, m in zip(group_indices, this_group):
            x_ticks[i] = m

    print(f"We have in tatol {len(x_ticks)} models")
    sns.despine()
    plt.ylim(0)
    plt.xticks(xs, x_ticks, rotation=90, ha="center")
    plt.xlim(-1, len(xs))
    plt.ylabel("Whole-brain alignment (normalised R)")

    plt.savefig(f"{FIGURE_DIR}/f1_score_bar-details-{args.region}-{args.clip_duration}.svg", bbox_inches="tight", dpi=300)
    plt.close()


    # Figure: box plot
    plt.subplots(figsize=(5*SCALE, 4))
    # box plot of groups
    for i, (group, mean, std, c) in enumerate(zip(GROUPS, means, stds, cs)):
        if group == "Others":
            continue
        sns.stripplot(x=i, y=mean, edgecolor="white", linewidth=.5, jitter=0.1, color=c, size=6)

        # print the top two model names in each group
        top_two = np.argsort(mean)[-2:]
        for j, (m, m_mean) in enumerate(zip(np.array(group_models[group])[top_two], np.array(mean)[top_two])):
            print(f"{group} {j+1}: {m} {m_mean}")

        plt.boxplot(mean, positions=[i], showfliers=False, widths=0.4, zorder=100)
    plt.xticks(range(len(GROUPS)), GROUPS, rotation=45, ha="right")
    plt.ylim(0, 0.55)
    plt.ylabel("Whole-brain alignment (normalised R)")

    sns.despine()

    plt.savefig(f"{FIGURE_DIR}/f1_score_bar-{args.region}-{args.clip_duration}.svg", bbox_inches="tight", dpi=300)
    plt.close()
                        

if __name__ == "__main__":
    args = get_args(
        ("--region", dict(default="Whole_Brain", type=str, help="data type to plot")),
    )
    main(args)
