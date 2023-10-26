using Random

# Simple random number password generator, good enough for my use.

function generate_password(len::Int)
    # Define the character sets
    uppercase_letters = 'A':'Z'
    lowercase_letters = 'a':'z'
    numbers = '0':'9'
    symbols = "!@#%^&*()-_=+[]{};:,.<>/?"

    all_chars = [uppercase_letters; lowercase_letters; numbers]
    ran_syms = rand(1:len-8) # Not more syms then total length - 8
    password_symbols = [rand(symbols) for _ in 1:ran_syms] # creates the random symbols 
    password_chars = [rand(all_chars) for _ in 1:(len-ran_syms)] # random characters and nums 

    password = shuffle!([password_symbols; password_chars]) # shuffles all the information
    return join(password) # Puts it into a string 
end

# Example usage
password = generate_password(20)  # Generate a 20-character
println("length = ", length(password), ": $password")

password = nothing #Sets ram var to zero, becouse why not.