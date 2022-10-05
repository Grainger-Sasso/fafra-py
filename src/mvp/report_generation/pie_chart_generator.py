import os
import time
import json
from matplotlib import pyplot as plt

from src.mvp.fafra_path_handler import PathHandler


class SKDHPlotGenerator:
    def __init__(self):
        pass

    def gen_skdh_plots(self, path_handler: PathHandler):
        # Build output path
        output_path = path_handler.ra_report_subcomponents_folder
        skdh_results_path = path_handler.skdh_pipeline_results_file
        # Read in skdh results
        skdh_results_data = self.read_json(skdh_results_path)
        # Generate pie charts
        act_pie_path, sleep_pie_path = self.gen_pie_charts(skdh_results_data, output_path, path_handler)
        # Generate activity heat maps
        return act_pie_path, sleep_pie_path

    def create_output_path(self, path_handler):
        base_path = path_handler.ra_model_folder
        folder_path = os.path.join(base_path, 'report_subcomponents')
        pass

    def gen_pie_charts(self, skdh_results_data, output_path, path_handler):
        # Grab activity and sleep data
        activity_data = skdh_results_data['act_metrics']
        sleep_data = skdh_results_data['sleep_metrics']
        # Create pie charts for activity and sleep
        act_fig, act_plot = self.create_act_pie_chart(activity_data)
        sleep_fig, sleep_plot = self.create_sleep_pie_chart(sleep_data)
        metric_file_name = 'model_input_metrics_' + time.strftime("%Y%m%d-%H%M%S") + '.json'
        act_plot_path = os.path.join(output_path, str('activity_pie_chart_' + time.strftime("%Y%m%d-%H%M%S") + '.png'))
        sleep_plot_path = os.path.join(output_path, str('sleep_pie_chart_' + time.strftime("%Y%m%d-%H%M%S") + '.png'))
        path_handler.ra_report_act_chart_file = act_plot_path
        path_handler.ra_report_sleep_chart_file = sleep_plot_path
        act_fig.savefig(act_plot_path, bbox_inches='tight')
        sleep_fig.savefig(sleep_plot_path, bbox_inches='tight')
        return act_plot_path, sleep_plot_path

    def create_act_pie_chart(self, act_data):
        total_time_min = act_data['N hours'][0] * 60.0
        wear_time_min = act_data['N wear hours'][0] * 60.0
        no_data_min = total_time_min - wear_time_min
        sed_time_min = act_data['wake sed 5s epoch [min]'][0]
        light_time_min = act_data['wake light 5s epoch [min]'][0]
        mod_time_min = act_data['wake mod 5s epoch [min]'][0]
        vig_time_min = act_data['wake vig 5s epoch [min]'][0]
        act_times = [
            sed_time_min,
            light_time_min,
            mod_time_min,
            vig_time_min
        ]
        labels = [
            'Sedentary',
            'Light Activity',
            'Moderate Activity',
            'Vigorous Activity'
        ]
        explode = [0.1, 0.1, 0.1, 0.1]
        colors = ['gainsboro', 'skyblue', 'royalblue', 'green']
        if no_data_min > 0.0:
            act_times.append(no_data_min)
            labels.append('No data')
            explode.append(0.1)
            colors.append('silver')
        act_times = [time / total_time_min for time in act_times]
        new_labels = []
        for label, value in zip(labels, act_times):
            new_labels.append(label + ' - ' + str(round(value * 100.0, 1)) + '%')
        fig1, ax1 = plt.subplots()
        # fig1.set_size_inches(5, 3)
        ax1.pie(act_times, startangle=90, explode=explode, shadow=True, colors=colors)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax1.legend(labels=new_labels, loc='lower center', prop={'size': 20},
                   bbox_to_anchor=(0.5, -0.2), ncol=2, fancybox=True, shadow=True)
        return fig1, ax1

    def create_sleep_pie_chart(self, sleep_data):
        sleep_percent = sleep_data['percent time asleep'][0]
        non_sleep_percent = 100.0 - sleep_percent
        percentages = [
            sleep_percent
        ]
        labels = [
            'Time asleep'
        ]
        colors = [
            'slateblue'
        ]
        explode = [
            0.1
        ]
        if non_sleep_percent > 0.0:
            percentages.append(non_sleep_percent)
            labels.append('Time awake')
            colors.append('darkslateblue')
            explode.append(0.1)
        new_labels = []
        for label, value in zip(labels, percentages):
            new_labels.append(label + ' - ' + str(round(value, 1)) + '%')
        fig1, ax1 = plt.subplots()
        ax1.pie(percentages, startangle=90, explode=explode, shadow=True, colors=colors, textprops={'fontsize': 16})
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax1.legend(labels=new_labels, loc='best')
        return fig1, ax1

    def read_json(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data


def main():
    skdh_path = '/home/grainger/Desktop/skdh_testing/ml_model/input_metrics/skdh/skdh_results_20220815-171703.json'
    output_path = '/home/grainger/Desktop/skdh_testing/fafra_results/reports/pie_charts/'
    pcg = SKDHPlotGenerator()
    pcg.gen_skdh_plots(skdh_path, output_path)


if __name__ == '__main__':
    main()

