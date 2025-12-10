import numpy as np
import scipy.special as sp
import yaml
import json
import matplotlib.pyplot as plt


class SphereRCS:
    def __init__(self, diameter: float):
        self.diameter = float(diameter)
        self.radius = self.diameter / 2.0

    def calculate_rcs(self, frequency: float) -> float:
        c = 3e8
        k = 2 * np.pi * float(frequency) / c
        r = self.radius
        ka = k * r

        if ka == 0:
            return 0.0

        n_max = int(ka + 10)
        if n_max < 10:
            n_max = 10

        total_sum = 0.0 + 0.0j

        for n in range(1, n_max + 1):
            jn_ka = sp.spherical_jn(n, ka)
            jn_ka_prime = sp.spherical_jn(n, ka, derivative=True)
            yn_ka = sp.spherical_yn(n, ka)
            yn_ka_prime = sp.spherical_yn(n, ka, derivative=True)

            hn_ka = jn_ka + 1j * yn_ka
            hn_ka_prime = jn_ka_prime + 1j * yn_ka_prime

            an = jn_ka / hn_ka
            bn = (ka * jn_ka_prime) / (ka * hn_ka_prime)

            term = (-1)**n * (n + 0.5) * (bn - an)
            total_sum += term

        wavelength = c / float(frequency)
        rcs = (wavelength ** 2 / np.pi) * np.abs(total_sum)**2

        return rcs


class ResultWriter:
    @staticmethod
    def write_json_format3(frequencies: list, rcs_values: list, filename: str):
        c = 3e8
        wavelengths = [c / float(f) for f in frequencies]

        data = {
            "freq": [float(f) for f in frequencies],
            "lambda": [float(w) for w in wavelengths],
            "rcs": [float(r) for r in rcs_values]
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Результаты сохранены в файл: {filename}")


class RCSCalculator:
    def __init__(self, input_file: str = "task_rcs_02.yaml"):
        self.input_file = input_file
        self.variants = {}
        self.load_variants()

    def load_variants(self):
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            # ВОТ ТУТ ГЛАВНОЕ ИЗМЕНЕНИЕ
            self.variants = data["data"]

            print(f"Успешно загружено {len(self.variants)} вариантов из файла {self.input_file}")

        except Exception as e:
            print(f"Ошибка при загрузке файла {self.input_file}: {e}")
            raise

    def print_available_variants(self):
        print("\nДоступные варианты:")
        print("№\tДиаметр (м)\t\tfmin (Гц)\t\tfmax (Гц)")
        print("-" * 70)

        for number, v in self.variants.items():
            print(f"{number}\t{float(v['D']):.6f}\t\t{float(v['fmin']):.2e}\t\t{float(v['fmax']):.2e}")

    def calculate_for_variant(self, variant_number: int, num_points: int = 1000):
        if variant_number not in self.variants:
            raise ValueError(f"Вариант {variant_number} не найден")

        variant = self.variants[variant_number]

        D = float(variant['D'])
        fmin = float(variant['fmin'])
        fmax = float(variant['fmax'])

        print(f"\nРасчет для варианта {variant_number}:")
        print(f"Диаметр сферы: {D} м")
        print(f"Диапазон частот: {fmin:.2e} - {fmax:.2e} Гц")
        print(f"Количество точек расчета: {num_points}")

        sphere = SphereRCS(D)
        frequencies = np.logspace(np.log10(fmin), np.log10(fmax), num_points)

        rcs_values = []
        for freq in frequencies:
            rcs_values.append(sphere.calculate_rcs(freq))

        return frequencies, rcs_values, variant

    def plot_results(self, frequencies: list, rcs_values: list, variant_number: int):
        variant = self.variants[variant_number]

        plt.figure(figsize=(12, 8))
        plt.semilogx(frequencies, rcs_values, 'b-', linewidth=2)
        plt.xlabel('Частота, Гц', fontsize=12)
        plt.ylabel('ЭПР, м²', fontsize=12)
        plt.title(f'ЭПР идеально проводящей сферы (D = {variant["D"]} м)', fontsize=14)
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()

        filename = f"sphere_rcs_variant_{variant_number}.png"
        plt.savefig(filename, dpi=300)
        plt.show()

        print(f"График сохранён как {filename}")

    def save_results(self, frequencies, rcs_values, variant_number):
        filename = f"rcs_results_variant_{variant_number}.json"
        ResultWriter.write_json_format3(frequencies, rcs_values, filename)
        return filename


def main():
    try:
        calculator = RCSCalculator("task_rcs_02.yaml")
        calculator.print_available_variants()

        variant_number = int(input("\nВведите номер варианта для расчета: "))

        frequencies, rcs_values, variant = calculator.calculate_for_variant(variant_number, num_points=500)
        calculator.plot_results(frequencies, rcs_values, variant_number)
        output = calculator.save_results(frequencies, rcs_values, variant_number)

        print("\nГотово!")

    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
