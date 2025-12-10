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

        total_sum = 0.0

        for n in range(1, n_max + 1):
            jn_ka = sp.spherical_jn(n, ka)
            jn_ka_prime = sp.spherical_jn(n, ka, derivative=True)
            yn_ka = sp.spherical_yn(n, ka)
            yn_ka_prime = sp.spherical_yn(n, ka, derivative=True)

            hn_ka = jn_ka + 1j * yn_ka
            hn_ka_prime = jn_ka_prime + 1j * yn_ka_prime

            an = jn_ka / hn_ka
            bn = (ka * jn_ka_prime) / (ka * hn_ka_prime)

            term = (-1) ** n * (n + 0.5) * (bn - an)
            total_sum += term

        wavelength = c / float(frequency)
        rcs = (wavelength ** 2 / np.pi) * np.abs(total_sum) ** 2

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
        self.variants = []
        self.load_variants()

    def load_variants(self):
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                self.variants = data['data']

            for variant_data in self.variants:
                variant = variant_data['variant']
                variant['D'] = float(variant['D'])
                variant['fmin'] = float(variant['fmin'])
                variant['fmax'] = float(variant['fmax'])

            print(f"Успешно загружено {len(self.variants)} вариантов из файла {self.input_file}")

        except Exception as e:
            print(f"Ошибка при загрузке файла {self.input_file}: {e}")
            raise

    def print_available_variants(self):
        print("\nДоступные варианты:")
        print("№\tДиаметр (м)\t\tfmin (Гц)\t\tfmax (Гц)")
        print("-" * 70)
        for variant_data in self.variants:
            v = variant_data['variant']
            print(f"{v['number']}\t{v['D']:.6f}\t\t{v['fmin']:.2e}\t\t{v['fmax']:.2e}")

    def calculate_for_variant(self, variant_number: int, num_points: int = 1000):
        variant = None
        for v in self.variants:
            if v['variant']['number'] == variant_number:
                variant = v['variant']
                break

        if variant is None:
            raise ValueError(f"Вариант {variant_number} не найден")

        D = variant['D']
        fmin = variant['fmin']
        fmax = variant['fmax']

        print(f"\nРасчет для варианта {variant_number}:")
        print(f"Диаметр сферы: {D} м")
        print(f"Диапазон частот: {fmin:.2e} - {fmax:.2e} Гц")
        print(f"Количество точек расчета: {num_points}")

        sphere = SphereRCS(D)
        frequencies = np.logspace(np.log10(fmin), np.log10(fmax), num_points)
        rcs_values = []
        for i, freq in enumerate(frequencies):
            rcs = sphere.calculate_rcs(freq)
            rcs_values.append(rcs)

        return frequencies, rcs_values, variant

    def plot_results(self, frequencies: list, rcs_values: list, variant: dict):
        plt.figure(figsize=(12, 8))
        plt.semilogx(frequencies, rcs_values, 'b-', linewidth=2)
        plt.xlabel('Частота, Гц', fontsize=12)
        plt.ylabel('ЭПР, м²', fontsize=12)
        plt.title(f'ЭПР идеально проводящей сферы (D = {variant["D"]} м)', fontsize=14)
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.figtext(0.02, 0.02,
                    f'Диапазон частот: {variant["fmin"]:.2e} - {variant["fmax"]:.2e} Гц\n'
                    f'Диаметр сферы: {variant["D"]} м',
                    fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        plt.tight_layout()
        plt.savefig(f'sphere_rcs_variant_{variant["number"]}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_results(self, frequencies: list, rcs_values: list, variant_number: int):
        filename = f"rcs_results_variant_{variant_number}.json"
        ResultWriter.write_json_format3(frequencies, rcs_values, filename)
        return filename

def main():
    try:
        calculator = RCSCalculator("task_rcs_02.yaml")
        calculator.print_available_variants()

        try:
            variant_number = int(input("\nВведите номер варианта для расчета: "))
        except ValueError:
            print("Ошибка: введите целое число")
            return

        variant_exists = any(v['variant']['number'] == variant_number for v in calculator.variants)
        if not variant_exists:
            print(f"Ошибка: вариант {variant_number} не найден в файле")
            return

        frequencies, rcs_values, variant = calculator.calculate_for_variant(variant_number, num_points=500)
        calculator.plot_results(frequencies, rcs_values, variant)
        output_file = calculator.save_results(frequencies, rcs_values, variant_number)

        print(f"\nРезультаты для варианта {variant_number}:")
        print(f"Диаметр сферы: {variant['D']} м")
        print(f"Диапазон частот: {variant['fmin']:.2e} - {variant['fmax']:.2e} Гц")
        print(f"Количество точек расчета: {len(frequencies)}")
        print(f"Минимальное значение ЭПР: {min(rcs_values):.2e} м²")
        print(f"Максимальное значение ЭПР: {max(rcs_values):.2e} м²")
        print(f"Файл с результатами: {output_file}")

        print("\nПервые 5 строк результатов из JSON файла:")
        with open(output_file, 'r') as f:
            data = json.load(f)
            for i in range(min(5, len(data['freq']))):
                print(f"Частота: {data['freq'][i]:.6e} Гц, "
                      f"Длина волны: {data['lambda'][i]:.6e} м, "
                      f"ЭПР: {data['rcs'][i]:.6e} м²")

        print(f"\nРасчет завершен для варианта {variant_number}")

    except FileNotFoundError:
        print("Ошибка: файл task_rcs_02.yaml не найден")
        print("Убедитесь, что файл находится в той же папке, что и скрипт")
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()