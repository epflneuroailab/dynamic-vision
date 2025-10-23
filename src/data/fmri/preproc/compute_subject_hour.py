from .. import data_store

total = 0
for data_name in [
    "savasegal2023-fmri-defeat",
    "savasegal2023-fmri-growth",
    "savasegal2023-fmri-iteration",
    "savasegal2023-fmri-lemonade",
    "keles2024-fmri",
    "berezutskaya2021-fmri",
    "mcmahon2023-fmri",
    "lahner2024-fmri",
]:

    data = data_store.load(data_name)
    subject_hour = data.sizes["time_bin"] * data.sizes["presentation"] / 3600
    total += subject_hour
    print(f"{data_name}: {subject_hour:.2f} hours")

print(f"Total: {total:.2f} hours")