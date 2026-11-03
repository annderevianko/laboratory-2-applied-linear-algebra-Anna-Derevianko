import numpy as np

def encrypt_message(message, key_matrix):
    message_vector = np.array([ord(char) for char in message])
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)),
                                     np.linalg.inv(eigenvectors))
    encrypted_vector = np.dot(diagonalized_key_matrix, message_vector)

    return encrypted_vector


def decrypt_message(encrypted_message, key_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    inverse_eigenvalues = 1 / eigenvalues
    inverse_diagonalized_matrix = eigenvectors @ np.diag(inverse_eigenvalues) @ np.linalg.inv(eigenvectors)
    decrypted_message = inverse_diagonalized_matrix @ encrypted_message

    answear = ''
    for i in decrypted_message:
        letter = chr(int(np.round(np.real(i))))
        answear = answear + letter

    return answear


message = "Hello World"
key_matrix = np.random.randint(0, 256, (len(message), len(message)))
encrypted_message = encrypt_message(message, key_matrix)
decrypted_message = decrypt_message(encrypted_message, key_matrix)
print(encrypted_message)
print(decrypted_message)