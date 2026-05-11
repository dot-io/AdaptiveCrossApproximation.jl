
# index CSC format
function kernel_index!(result, row_indices, col_ptrs, val, a, b, na, nb)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if (i <= na)
        for j in 1:nb
            row = a[i] - 1
            col = b[j]
            value = 0.0
            for k in col_ptrs[col]:(col_ptrs[col + 1] - 1)
                if row_indices[k] == row
                    value = val[k]
                    break
                end
            end
            result[i, j] = value
        end
    end
    return nothing
end

# index CSR format
function kernel_index!(result, row_ptrs, col_indices, val, a, b, na, nb)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if (i <= na)
        for j in 1:nb
            row = a[i]
            col = b[j] - 1
            value = 0.0
            for k in row_ptrs[row]:(row_ptrs[row + 1] - 1)
                if col_indices[k] == col
                    value = val[k]
                    break
                end
            end
            result[i, j] = value
        end
    end
    return nothing
end
