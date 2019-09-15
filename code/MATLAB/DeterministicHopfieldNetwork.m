classdef DeterministicHopfieldNetwork < handle
    properties (Access = private)
        weights = [];
        patterns = [];
        diagonal_weights_zero = true;
    end
    
    methods
        function set_diagonal_weights_to_zero(obj,diagonal_weights_to_zero)
            obj.diagonal_weights_zero = diagonal_weights_to_zero;
        end
        
        function set_patterns(obj,patterns)
            obj.patterns = patterns;
        end
        
        function patterns = get_patterns(obj)
            patterns = obj.patterns;
        end
        
        function generate_weights(obj)
            [n_bits,n_patterns] = size(obj.patterns);
            obj.weights = zeros(n_bits,n_bits);
            for pattern_i = 1:n_patterns
                obj.weights = obj.weights + obj.patterns(:,pattern_i)*obj.patterns(:,pattern_i)';
            end
            obj.weights = obj.weights/n_patterns;
            
            if obj.diagonal_weights_zero
                zero_diagonal_matrix = ~eye(size(obj.weights));
                obj.weights = obj.weights .* zero_diagonal_matrix;
            end
        end
        
        function weights = get_weights(obj)
            weights = obj.weights;
        end
        
        function updated_pattern = update_bit_of_pattern(obj,pattern,bit_index)
            weights_i = obj.weights(bit_index,:);
            b = weights_i*pattern;
            updated_bit = obj.sign_zero_returns_one(b);
            updated_pattern = pattern;
            updated_pattern(bit_index) = updated_bit;
        end
    end
    
    methods (Static, Access = private)
        function sign_of_value = sign_zero_returns_one(value)
            sign_of_value = sign(value);
            if sign_of_value == 0
                sign_of_value = 1;
            end
        end
    end
end