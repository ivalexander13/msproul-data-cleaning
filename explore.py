# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Exploratory Callink Data Analysis for Cleanup
# Explore the data and write the cleanup code for a pipeline. 
# For every column, if cleanup is needed, write a function called cleanup_colname(df) that accepts a df and returns a df. Collect these into the SET cleanup_fn.
# 
# Always assume that the column values can be NaN!
# %% [markdown]
# ## Imports and setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
sns.set(style = "whitegrid", 
        color_codes = True)


# %%
og_data = pd.read_csv('all-rso.csv')
copy1 = og_data.copy()


# %%
# List of cleanup functions! Use this at the end.
cleanup_fn = set()


# %%
og_data.head()

# %% [markdown]
# ## High Level Inspection

# %%
cols = copy1.columns.values
cols

# %% [markdown]
# How many missing values are there per column?

# %%
counts_blank = [copy1[col].notnull().sum() for col in cols]

plt.figure(figsize=(8,9));
plt.barh(width=counts_blank, y=cols);
plt.xlim(0, len(copy1));
plt.yticks(cols)
plt.title("Number of non-NaN's in the original data.");

# %% [markdown]
# ## Per Column Cleaning
# %% [markdown]
# ## 1. ID
# %% [markdown]
# Is ID unique for each RSO?

# %%
np.all(pd.unique(copy1["id"]) == copy1["id"])


# %%
sorted_id = copy1.sort_values(by="id", ascending=True)
sorted_id.head()

# %% [markdown]
# The ID seems to be in order based on the RSO's registration date. Let's check if it's incremental.

# %%
smallest_val = sorted_id["id"].iloc[0]
largest_val = sorted_id["id"].iloc[-1]
largest_val == smallest_val + len(sorted_id) -1

# %% [markdown]
# No cleanup needed.
# %% [markdown]
# ## 2. Fullname
# %% [markdown]
# What is the longest and shortest name?

# %%
max_num = np.argmax(copy1.fullname.apply(len))
max_name = copy1.loc[max_num,'fullname']

min_num = np.argmin(copy1.fullname.apply(len))
min_name = copy1.loc[min_num,'fullname']

# print(f"Longest name is: \n   {max_name} \n at {len(max_name)}.")
# print()
# print(f"Shortest name is: \n   {min_name} \n at {len(min_name)}.")

# %% [markdown]
# No cleanup needed.
# %% [markdown]
# ## 3. Keyname

# %%
max_num = np.argmax(copy1.keyname.apply(len))
max_name = copy1.loc[max_num,'keyname']

min_num = np.argmin(copy1.keyname.apply(len))
min_name = copy1.loc[min_num,'keyname']

# print(f"Longest keyname is: \n   {max_name} \n at {len(max_name)}.")
# print()
# print(f"Shortest keyname is: \n   {min_name} \n at {len(min_name)}.")

# %% [markdown]
# No cleanup needed.
# %% [markdown]
# ## 4. rso_email
# 
# %% [markdown]
# How many different domains do the emails belong to?

# %%
def find_email_domain(email):
    regex = '(?<=@).*?([^.]+\.[\w]+)$'
    found_list = re.findall(regex, email)
    if len(found_list) != 0:
        return found_list[-1]

email_series = copy1.rso_email.dropna().apply(find_email_domain, convert_dtype=True)
email_series = email_series.value_counts()
email_series

# %% [markdown]
# There's one in this field, index 884, whose email is 'mixedatberkeley.com'. Clearly invalid. Filter! Also, although the vast majority are either gmail or berkeley.edu, there are an overwhelming number of custom email hosting.

# %%
copy2 = og_data.copy()
def cleanup_rso_email(df):
    def is_valid(email):
        re_valid_email = "(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
        if email is None or email is np.nan:
            return np.nan
        elif re.match(re_valid_email, email) is not None:
            return email.lower()
        else:
            return np.nan

    df.rso_email = df.rso_email.apply(is_valid, convert_dtype=True)
    return df

cleanup_fn.add(cleanup_rso_email)

#cleanup_rso_email(copy2)

# %% [markdown]
# ## 5. Description + cleanup
# Use this code for cleanup!

# %%
description_series = copy1.description 

def longtext_cleanup(text):
    found = ' '.join(re.sub("(?i)<.?[^>]*>", ' ', text).split())
    # found = ' '.join(re.sub("&nbsp;", ' ', found).split())
    # found = ' '.join(re.sub("&lt;", '<', found).split())
    # found = ' '.join(re.sub("&gt;", '>', found).split())
    # found = ' '.join(re.sub("&amp;", '&', found).split())
    # found = ' '.join(re.sub("&quot;", '"', found).split())
    # found = ' '.join(re.sub("&apos;", '\'', found).split())
    # found = ' '.join(re.sub("&cent;", '¢', found).split())
    # found = ' '.join(re.sub("&pound;", '£', found).split())
    # found = ' '.join(re.sub("&yen;", '¥', found).split())
    # found = ' '.join(re.sub("&euro;", '€', found).split())
    # found = ' '.join(re.sub("&copy;", '©', found).split())
    # found = ' '.join(re.sub("&reg;", '®', found).split())
    if found != '' and found is not None:
        return found
description_series = copy1.description.dropna().apply(longtext_cleanup)
description_series

# %% [markdown]
# Ignore the HTML entities. They might come in handy when transferring between media.

# %%
copy2 = og_data.copy()
def cleanup_description(df):
    def is_valid(text):
        regex = "(?i)<.?[^>]*>"
        if text is None or text is np.nan or len(text) == 0:
            return np.nan
        else:
            return ' '.join(re.sub(regex, ' ', text).split())

    df.description = df.description.apply(is_valid, convert_dtype=True)
    return df

cleanup_fn.add(cleanup_description)

#cleanup_description(copy2).description.dropna()

# %% [markdown]
# ## 6. Summary + cleanup

# %%
summary_series = copy1.summary.dropna().apply(longtext_cleanup)
summary_series

# %% [markdown]
# There are plenty of placeholder text! See which ones are the most common and filter them out.

# %%
summary_fillers = summary_series.value_counts()


# %%
def remove_summary_filler(text):
    if (
        text == '.' or
        text == '...' or
        text == '---' or
        text == 'tbd' or
        text == '[To be filled out by members of the organization]' or
        text == '#NAME?' or
        text == '(fill)' or
        text == 'To be entered' or
        len(text) < 8
    ):
        return None # None or empty string?
    else:
        return text

summary_remove_filler = summary_series.apply(remove_summary_filler)
a = summary_remove_filler.value_counts()


# %%
copy2 = og_data.copy()
def cleanup_summary(df):
    def return_valid(val):
        if val is None or val is np.nan or len(val) == 0:
            return np.nan
        else:
            return remove_summary_filler(val)

    df.summary = df.summary.apply(remove_summary_filler, convert_dtype=True)
    return df

cleanup_fn.add(cleanup_summary)

cleanup_summary(copy2).summary.dropna()

# %% [markdown]
# ## 7. Start Date

# %%
copy1.start_date.value_counts()

# %% [markdown]
# Highly unlikely 357 RSOs registered at the same time in 2013. Filter that out along with 1969. Also I think anything more specific than year is irrelevant.
# Also, if start date is unavailable, use changestatus date if it exists. I dont think there's a correlation between the presence of startdate and presence of changestatusdate.

# %%
copy2 = og_data.copy()
def cleanup_start_date(df):
    def return_valid(val):
        if (
            val is None 
            or val is np.nan 
            or len(val) == 0
            or val == '2013-07-29T00:00:00+00:00' 
            or val == '1969-12-31T00:00:00+00:00'
        ):
            return np.nan
        else: 
            return val[0:4]

    df.start_date = df.start_date.apply(return_valid, convert_dtype=True)
    df.start_date = df.start_date.fillna(df.status_change_date)
    df.start_date = df.start_date.apply(return_valid, convert_dtype=True)
    return df

cleanup_fn.add(cleanup_start_date)

cleanup_start_date(copy2).start_date.dropna()

# %% [markdown]
# Just for fun!

# %%
year_df = cleanup_start_date(copy2).start_date.dropna().value_counts().sort_index()
plt.plot(year_df.index, year_df);
plt.yticks(range(0, max(year_df) + 1, 10))
plt.title("Number of RSO registrations by Year.\n(Only 2/3 accounted for)");
plt.ylabel("Number of RSOs");
plt.xlabel("Year");

# %% [markdown]
# ## 8. The Socials

# %%
social_colnames = [colname for colname in copy1.columns if colname.startswith('social_')]
social_colnames

# %% [markdown]
# Based on manual inspection, there is no need for cleanup.
# %% [markdown]
# # Closing
# %% [markdown]
# We have these functions that will clean up the corresponding columns. Add one last final function to finish it off. 
# 
# Final export: cleanup_pipeline(df)!!!

# %%
cleanup_fn


# %%
# copy2 = og_data.copy()
# for fn in cleanup_fn:
#     copy2 = fn(copy2)
# copy2


# %%
def cleanup_pipeline(df):
    df = df.copy()
    for fn in cleanup_fn:
        df = fn(df)
    return df


# %%
# Save to csv
cleanup_pipeline(og_data.copy()).to_csv(r'/home/ivalexander13/msproul-local/data-cleaning/processed_all_rso.csv', index=False)


# %%
# See the missing data graph again, post-processing.
copy2 = cleanup_pipeline(og_data.copy())
cols = copy2.columns.values
counts_blank = [copy2[col].notnull().sum() for col in cols]

plt.figure(figsize=(8,9));
plt.barh(width=counts_blank, y=cols);
plt.xlim(0, len(copy2));
plt.yticks(cols)
plt.title("Number of non-NaN's in the processed data.");


# %%



