from statsmodels.stats.contingency_tables import Table
from scipy.stats import fisher_exact

def fisher_exact_test(table: list):
    """
    Perform Fisher's exact test on a 2x2 contingency table.
    Args:
        table (list): A 2x2 contingency table.
    Returns:
        float: The p-value from Fisher's exact test.
    """
    tbl = Table(table)
    return (tbl.test_nominal_association().pvalue)  # Fisher's exact test p-value


def fisher_exact_scipy_test(table: list, alternative=None, method=None):
    """
    Perform Fisher's exact test on a 2x2 contingency table using scipy, othersize alternative must be None.
    Args:
        table (list): A 2x2 contingency table.
        alternative (str): The alternative hypothesis ('two-sided', 'less', 'greater').
        method (str): The method to use ('exact', 'approximate').
    Returns:
        float: The p-value from Fisher's exact test.
    """

    _, p_value = fisher_exact(table, alternative)

    print("Approximate Fisher p-value:", p_value)
    return p_value

if __name__ == "__main__":
    # Example 2x2 table
    table = [[10, 5], [20, 15]]

    # Call the function
    p_value = fisher_exact_test(table)

    print("Fisher's exact test p-value:", p_value)

    table = [[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 1, 0]]
    p_value = fisher_exact_test(table)
    print("Fisher's exact test p-value:", p_value)

    table = [[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 1, 0]]
    p_value = fisher_exact_scipy_test(table)
    print("Fisher's exact test p-value:", p_value)

    table = [[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 1, 0]]
    p_value = fisher_exact_scipy_test(table,  method='exact')
    print("Fisher's exact test p-value:", p_value)

    table = [[1, 0], [0, 1]]
    p_value = fisher_exact_scipy_test(table, alternative='greater', method='exact')
    print("Fisher's exact test p-value:", p_value)