function tests = test_generate_n_patterns
    tests = functiontests(localfunctions);
end

function test_correct_size(test_case)
    n_bits = 100;
    n_patterns = 10;
    patterns = generate_n_patterns(n_bits,n_patterns);
    actual_solution = size(patterns);
    expected_solution = [n_bits,n_patterns];
    verifyEqual(test_case,actual_solution,expected_solution);
end

function test_all_values_are_minus_1_or_1(test_case)
    n_bits = 120;
    n_patterns = 12;
    patterns = generate_n_patterns(n_bits,n_patterns);
    unique_values = unique(unique(patterns));
    actual_solution = all(unique_values == [-1;1]);
    expected_solution = true;
    verifyEqual(test_case,actual_solution,expected_solution);
end