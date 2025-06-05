import pickle


class HyperSpheres:
    def __init__(self, x_data, spheres_hs, sphere_HS):
        self.x_data = x_data
        self.spheres_hs = spheres_hs
        self.sphere_HS = sphere_HS

    def save_to_file(self, filename):
        data = {
            'x_data': self.x_data,
            'spheres_hs': self.spheres_hs,
            'sphere_HS': self.sphere_HS,
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {filename}")

    @classmethod
    def load_from_file(cls, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return cls(**data)