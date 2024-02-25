# PandasAI 🐼

[![Release](https://img.shields.io/pypi/v/pandasai?label=Release&style=flat-square)](https://pypi.org/project/pandasai/)
[![CI](https://github.com/gventuri/pandas-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/gventuri/pandas-ai/actions/workflows/ci.yml/badge.svg)
[![CD](https://github.com/gventuri/pandas-ai/actions/workflows/cd.yml/badge.svg)](https://github.com/gventuri/pandas-ai/actions/workflows/cd.yml/badge.svg)
[![Coverage](https://codecov.io/gh/gventuri/pandas-ai/branch/main/graph/badge.svg)](https://codecov.io/gh/gventuri/pandas-ai)
[![Documentation Status](https://readthedocs.org/projects/pandas-ai/badge/?version=latest)](https://pandas-ai.readthedocs.io/en/latest/?badge=latest)
[![Discord](https://dcbadge.vercel.app/api/server/kF7FqH2FwS?style=flat&compact=true)](https://discord.gg/kF7FqH2FwS)
[![Downloads](https://static.pepy.tech/badge/pandasai)](https://pepy.tech/project/pandasai) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZnO-njhL7TBOYPZaqvMvGtsjckZKrv2E?usp=sharing)

PandasAI is a Python library that adds Generative AI capabilities to [pandas](https://github.com/pandas-dev/pandas), the popular data analysis and manipulation tool. It is designed to be used in conjunction with pandas, and is not a replacement for it.

<!-- Add images/pandas-ai.png -->

![PandasAI](images/pandas-ai.png?raw=true)

## 🔧 Quick install

```bash
pip install pandasai
```

## 🔍 Demo

Try out PandasAI in your browser:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZnO-njhL7TBOYPZaqvMvGtsjckZKrv2E?usp=sharing)

## 📖 Documentation

The documentation for PandasAI can be found [here](https://pandas-ai.readthedocs.io/en/latest/).

## 💻 Usage

> Disclaimer: GDP data was collected from [this source](https://ourworldindata.org/grapher/gross-domestic-product?tab=table), published by World Development Indicators - World Bank (2022.05.26) and collected at National accounts data - World Bank / OECD. It relates to the year of 2020. Happiness indexes were extracted from [the World Happiness Report](https://ftnnews.com/images/stories/documents/2020/WHR20.pdf). Another useful [link](https://data.world/makeovermonday/2020w19-world-happiness-report-2020).

PandasAI is designed to be used in conjunction with pandas. It makes pandas conversational, allowing you to ask questions to your data in natural language.

### Queries

For example, you can ask PandasAI to find all the rows in a DataFrame where the value of a column is greater than 5, and it will return a DataFrame containing only those rows:

```python
import pandas as pd
from pandasai import SmartDataframe

# Sample DataFrame
df = pd.DataFrame({
    "country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
    "gdp": [19294482071552, 2891615567872, 2411255037952, 3435817336832, 1745433788416, 1181205135360, 1607402389504, 1490967855104, 4380756541440, 14631844184064],
    "happiness_index": [6.94, 7.16, 6.66, 7.07, 6.38, 6.4, 7.23, 7.22, 5.87, 5.12]
})

# Instantiate a LLM
from pandasai.llm import OpenAI
llm = OpenAI(api_token="YOUR_API_TOKEN")

df = SmartDataframe(df, config={"llm": llm})
df.chat('Which are the 5 happiest countries?')
```

The above code will return the following:

```
6            Canada
7         Australia
1    United Kingdom
3           Germany
0     United States
Name: country, dtype: object
```

You can also ask PandasAI with Customerized LLM as well. Assume the API follows the OpenAI style (without token). For example, one could user [lm-studio](https://lmstudio.ai/) with `mixtral 8x7B` with this setting:

![lm-studio-settings](./images/lm-studio.png?raw=true)

```python
import pandas as pd
from pandasai import SmartDataframe

# Sample DataFrame
df = pd.DataFrame({
    "country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
    "gdp": [19294482071552, 2891615567872, 2411255037952, 3435817336832, 1745433788416, 1181205135360, 1607402389504, 1490967855104, 4380756541440, 14631844184064],
    "happiness_index": [6.94, 7.16, 6.66, 7.07, 6.38, 6.4, 7.23, 7.22, 5.87, 5.12]
})
from pandasai.llm import CustOpenAI

_host_url_ = "http://127.0.0.1"
_port_number_ = '1378'
_llm_version_ = 'v1' 
llm = CustOpenAI(api_base = f"{_host_url_}:{_port_number_}/{_llm_version_}", api_token = "null")

# T1:
llm.chat_completion('Hi')
# T2:
df_llm = SmartDataframe(df, config={"llm": llm})
df_llm.chat('Which are the 5 happiest countries?')
df_llm.chat('What is the sum of the GDPs of the 2 unhappiest countries?')
```

#### Query Examples (local mode)

The above code will return the following:

```
### Example: syntax Error:
'Unfortunately, I was not able to answer your question, because of the following error:\n\nNo code found in the response\n'

### Example: W/O syntax Error:
'The 5 happiest countries are: Canada, Australia, United Kingdom, Germany, United States'

### Example: Intent Identification Error:
#' `df_llm.chat('What is the sum of the GDPs of the 2 unhappiest countries?')`
#' It actually answers "What is the sum of the GDPs of the top 2 happiest countries?"
'The sum of the GDPs of the 2 unhappiest countries is: 53070779908096' 
```

Besides, it really depends on the quality of the question when the baseline LLM is not optimal. Check this eaxmple of WizardCoder-13B:

```python
## WRONG
In [3]: df_llm.chat('What is the sum of the GDPs of the 2 unhappiest countries?')
Out[3]: 'The sum of the GDPs of the 2 unhappiest countries is: 53070779908096'

## RIGHT
In [4]: df_llm.chat('What is the sum of the GDPs of the top 2 least happy countries?')
Out[4]: 'The sum of the GDPs of the top 2 least happy countries is: 19012600725504'
```

Of course, you can also ask PandasAI to perform more complex queries. For example, you can ask PandasAI to find the sum of the GDPs of the 2 unhappiest countries:

```python
df.chat('What is the sum of the GDPs of the 2 unhappiest countries?')
```

The above code will return the following:

```
19012600725504
```

### Charts

You can also ask PandasAI to draw a graph:

```python
df_llm.chat(
    "Plot the histogram of countries showing for each the gdp, using different colors for each bar",
)
```

![Chart](images/histogram-chart.png?raw=true)

You can save any charts generated by PandasAI by setting the `save_charts` parameter to `True` in the `PandasAI` constructor. For example, `PandasAI(llm, save_charts=True)`. Charts are saved in `./pandasai/exports/charts` .

>! Notice: the prompt varies according to the model you choose. The `OpenAI` style might fail on open-source LLMs. Users are encouraged to define their own prompts. 

### Multiple DataFrames

Additionally, you can also pass in multiple dataframes to PandasAI and ask questions relating them.

```python
import pandas as pd
from pandasai import SmartDatalake
from pandasai.llm import OpenAI

employees_data = {
    'EmployeeID': [1, 2, 3, 4, 5],
    'Name': ['John', 'Emma', 'Liam', 'Olivia', 'William'],
    'Department': ['HR', 'Sales', 'IT', 'Marketing', 'Finance']
}

salaries_data = {
    'EmployeeID': [1, 2, 3, 4, 5],
    'Salary': [5000, 6000, 4500, 7000, 5500]
}

employees_df = pd.DataFrame(employees_data)
salaries_df = pd.DataFrame(salaries_data)


llm = OpenAI()
dl = SmartDatalake([employees_df, salaries_df], config={"llm": llm})
dl.chat("Who gets paid the most?")
```

The above code will return the following:

```
Oh, Olivia gets paid the most.
```

You can find more examples in the [examples](examples) directory.

### ⚡️ Shortcuts

PandasAI also provides a number of shortcuts (beta) to make it easier to ask questions to your data. For example, you can ask PandasAI to `clean_data`, `impute_missing_values`, `generate_features`, `plot_histogram`, and many many more.

```python
# Clean data
df_llm.clean_data()

# Impute missing values
df_llm.impute_missing_values()

# Generate features
df_llm.generate_features()

# Plot histogram
df_llm.plot_histogram(column="gdp")
```

For the GDP historgam, the above code generates the historgraph accordingly.

![ex01-hist](./images/ex01-hist.png?raw=true)

The LLM (mixtral-7x8B) actually generates:


> Sure, I understand that the error is due to the `tempfile` module being used by the code generator but it's not in the whitelist of allowed modules for import. Here's the updated code without using `tempfile`:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Assuming dfs is already defined as a list of DataFrames
dfs[0].reset_index(drop=True, inplace=True)  # ensure index starts from 0
gdp_data = dfs[0]['gdp'].tolist()  # extract gdp column data to a list
plt.hist(gdp_data, bins='auto')  # plot histogram of the list
plt.title('GDP Distribution')
plt.xlabel('GDP ($ billions)')
plt.ylabel('Frequency')
plt.show()

result = {
    "type": "plot",
    "value": None
}
```
> In this updated code, I first reset the index of `dfs[0]` to ensure that it starts from 0. Then, I extract the 'gdp' column data into a list called `gdp_data`. Finally, I plot the histogram using `plt.hist(gdp_data, bins='auto')`, which directly takes in the list of GDP values instead of the DataFrame column object.
> 
> This should resolve the import error and still produce the desired histogram plot for the 'gdp' column data.

Learn more about the shortcuts [here](https://pandas-ai.readthedocs.io/en/latest/shortcuts/).

## 🔒 Privacy & Security

In order to generate the Python code to run, we take the dataframe head, we randomize it (using random generation for sensitive data and shuffling for non-sensitive data) and send just the head.

Also, if you want to enforce further your privacy you can instantiate PandasAI with `enforce_privacy = True` which will not send the head (but just column names) to the LLM.

## 🤝 Contributing

Contributions are welcome! Please check out the todos below, and feel free to open a pull request.
For more information, please see the [contributing guidelines](CONTRIBUTING.md).

After installing the virtual environment, please remember to install `pre-commit` to be compliant with our standards:

```bash
pre-commit install
```

## Contributors

[![Contributors](https://contrib.rocks/image?repo=gventuri/pandas-ai)](https://github.com/gventuri/pandas-ai/graphs/contributors)

## 📜 License

PandasAI is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements

- This project is based on the [pandas](https://github.com/pandas-dev/pandas) library by independent contributors, but it's in no way affiliated with the pandas project.
- This project is meant to be used as a tool for data exploration and analysis, and it's not meant to be used for production purposes. Please use it responsibly.
