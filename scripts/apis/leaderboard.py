from scripts.utils import *


def main(args):
    tmp = collect(args, types=args.data_types, models=args.models, exclude_pixels=False, silent=True)
    data = list(tmp[:-1])
    models = tmp[-1]

    for i, data_type in enumerate(args.data_types):
        if data_type == "fmri":
            data[i] = {"combined_fmri": data[i]}
            ceiling = get_ceiling(args)

    for datum, data_type in zip(data, args.data_types):
        for benchmark in datum:
            model_scores = []
            for model in models:
                model_score = datum[benchmark][model]
                if data_type == "fmri":
                    model_score = select_and_ceil(model_score, ceiling, region="Whole_Brain")
                model_scores.append(np.nanmean(model_score))
            best_indices = np.argsort(model_scores)[::-1]
            if args.top_k is not None:
                best_indices = best_indices[:args.top_k]
            best_models = [models[i] for i in best_indices]
            best_model_scores = np.array(model_scores)[best_indices]

            print('\n' + benchmark)
            for i, model in enumerate(best_models):
                print(f"{benchmark} {model}: {best_model_scores[i]:.4f}")


if __name__ == "__main__":
    args = get_args(
        ("--top_k", dict(default=None, type=int)),
        ("--data_types", dict(default=["fmri"], nargs='+', type=str)),
        ("--models", dict(default=ALL_MODELS, nargs='+', type=str))
    )
    main(args)

