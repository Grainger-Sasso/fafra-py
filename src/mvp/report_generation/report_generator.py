import os
import shutil
import numpy as np
import pandas as pd
import calendar
import json
from datetime import datetime
from fpdf import FPDF

import matplotlib.pyplot as plt
from matplotlib import rcParams

from src.mvp.report_generation.pie_chart_generator import SKDHPlotGenerator
from src.mvp.fafra_path_handler import PathHandler

rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
HEIGHT = 297
WIDTH = 210


class ReportGenerator:
    def __init__(self):
        self.margin_x = 20
        self.margin_y = 5
        self.rect_width = 170
        self.rect_height = 60
        self.current_x = 0
        self.current_y = 0

    def generate_report(self, path_handler: PathHandler, ra_results):
        # Build path to skdh_results_... that are parsed out; used for gait, act, and sleep
        skdh_results_path = path_handler.skdh_pipeline_results_file
        skdh_results = self.parse_json(skdh_results_path)
        # Build path to assessment info; used for testing information
        assessment_info_path = path_handler.assessment_info_file
        assessment_info_data = self.parse_json(assessment_info_path)
        # Build path to User info; used for demographic information
        user_info_path = path_handler.user_info_file
        user_info_data = self.parse_json(user_info_path)
        #
        pdf = PDF()
        # WRITE IN FRAMEWORK FOR THE WHOLE REPORT
        pdf.add_page()
        # Writes header
        pdf.set_font("helvetica", "", 16)

        # Key Layout Variables
        margin_x = 20
        margin_y = 5
        rect_width = 170
        rect_height = 60
        current_x = 0
        current_y = 0

        # From user profile path
        user_name = user_info_data['user_name']
        user_id = user_info_data['user_ID']
        # From test data path
        report_date = assessment_info_data['report_date']
        report_date = report_date[5:7] + '/' + report_date[8:] + '/' + report_date[0:4]
        report_generated = report_date
        assessment_ID = assessment_info_data['assessment_ID']
        recording_start = assessment_info_data['recording_start']
        recording_start = recording_start[5:7] + '/' + recording_start[8:] + '/' + recording_start[0:4]
        recording_end = assessment_info_data['recording_end']
        recording_end = recording_end[5:7] + '/' + recording_end[8:] + '/' + recording_end[0:4]
        collection_period = recording_start + '-' + recording_end
        # From fafra path
        fall_risk_score = self.set_fall_risk_value(ra_results)
        gait_speed = str(round(skdh_results['gait_metrics']['PARAM:gait speed: mean'], 2)) + ' m/s'
        cadence = str(round(skdh_results['gait_metrics']['PARAM:cadence: mean'], 2)) + '\n' + ' steps/min'
        steps_per_day = str(round(skdh_results['gait_metrics']['Bout Steps: sum'] / 2, 2))


        # Header
        current_y += margin_y
        pdf.set_fill_color(211, 211, 211)
        # Header rectangles and logo image
        pdf.rect(margin_x, current_y, rect_width, 30, style='DF')
        pdf.image("carapace_logo.jpg", margin_x + 6, current_y + 1, 28, 28)
        pdf.rect(margin_x + 45, current_y + 2, 120, 26, style='')
        # Required fields for header
        pdf.set_font("helvetica", "", 8)
        pdf.text(margin_x + 50, current_y + 7, "User Name: " + user_name)
        pdf.text(margin_x + 105, current_y + 7, "Report Generated: " + report_generated)
        pdf.text(margin_x + 50, current_y + 15, "User ID: " + user_id)
        pdf.text(margin_x + 105, current_y + 15, "Assessment ID: " + assessment_ID)
        pdf.text(margin_x + 50, current_y + 23, "Data Collection Period: " + collection_period)
        pdf.set_font("helvetica", "", 16)
        current_y += 30
        #####
        # Writes in the general activity and sleep sections
        pdf.set_fill_color(211, 211, 211)
        pdf.rect(x=20, y=255 - 2 * HEIGHT / 4, w=WIDTH - 40, h=60, style='DF')
        pdf.rect(x=20, y=245 - HEIGHT / 4, w=WIDTH - 40, h=60, style='DF')
        pdf.rect(x=40, y=265 - 2 * HEIGHT / 4, w=45, h=45, round_corners=True)
        pdf.rect(x=40, y=256 - HEIGHT / 4, w=45, h=45, round_corners=True)
        #####
        # Fills in the detail of the sleep and activity sections, adds images
        # Generate the pie charts, pass their paths to this list
        plt_gen = SKDHPlotGenerator()
        base_path = path_handler.risk_report_folder
        folder_path = os.path.join(base_path, 'report_subcomponents')
        path_handler.ra_report_subcomponents_folder = folder_path
        plt_gen.gen_skdh_plots(path_handler)
        image_list = []
        image_list.append(
            '/home/grainger/PycharmProjects/fafra-py/src/mvp/report_generation/Daily Activity Summary.png')
        image_list.append(path_handler.ra_report_act_chart_file)
        image_list.append(path_handler.ra_report_sleep_chart_file)
        pdf.print_page(image_list, skdh_results)
        # header
        # fall risk section
        # activity section
        # sleep section
        # WRITE IN THE PARAMETERIZED COMPONENTS
        # Fall Risk Report
        # Report Header
        current_y += margin_y
        pdf.rect(margin_x, current_y, rect_width, rect_height, style='DF')
        pdf.text(margin_x + 75, current_y + 10, "Fall Risk Report")
        pdf.set_font("helvetica", "", 13)
        pdf.rect(margin_x + 5, current_y + 15, w=50, h=40, round_corners=True)
        pdf.text(margin_x + 15, current_y + 20, "Fall Risk Score")
        # Set color for fall-risk rectangles
        if fall_risk_score == 'high':
            pdf.set_fill_color(255, 0, 0)
        if fall_risk_score == 'medium':
            pdf.set_fill_color(255, 255, 0)
        if fall_risk_score == 'low':
            pdf.set_fill_color(0, 255, 0)
        high_style = 'DF' if fall_risk_score == 'high' else 'D'
        medium_style = 'DF' if fall_risk_score == 'medium' else 'D'
        low_style = 'DF' if fall_risk_score == 'low' else 'D'
        pdf.rect(margin_x + 15, current_y + 22, w=30, h=8, style=high_style)
        pdf.text(margin_x + 24, current_y + 27, "HIGH")
        pdf.rect(margin_x + 15, current_y + 32, w=30, h=8, style=medium_style)
        pdf.text(margin_x + 21, current_y + 37, "MEDIUM")
        pdf.rect(margin_x + 15, current_y + 42, w=30, h=8, style=low_style)
        pdf.text(margin_x + 25, current_y + 47, "LOW")
        pdf.set_fill_color(211, 211, 211)
        # Fall risk indicators and its fields
        pdf.rect(margin_x + 80, current_y + 15, w=80, h=40, round_corners=True)
        pdf.text(margin_x + 100, current_y + 20, "Fall Risk Indicators")
        pdf.set_font("helvetica", "", 11)
        pdf.text(margin_x + 85, current_y + 30, "Gait Speed")
        pdf.text(margin_x + 107, current_y + 30, "|")
        pdf.text(margin_x + 90, current_y + 40, gait_speed)
        pdf.text(margin_x + 110, current_y + 30, "Cadence")
        pdf.text(margin_x + 127, current_y + 30, "|")
        pdf.text(margin_x + 115, current_y + 40, cadence)
        pdf.text(margin_x + 130, current_y + 30, "Steps per day")
        pdf.text(margin_x + 133, current_y + 40, steps_per_day)
        pdf.set_font("helvetica", "", 16)
        current_y += rect_height

        pdf.output('/home/grainger/Desktop/skdh_testing/fafra_results/reports/whole_report/SalesReport.pdf')

    def parse_json(self, json_file_path):
        # Read results JSON
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        return data

    def set_fall_risk_value(self, ra_results):
        prediction = ra_results['prediction']
        if int(prediction) == 0:
            fall_risk = 'low'
        elif int(prediction) == 1:
            fall_risk = 'medium'
        elif int(prediction) == 2:
            fall_risk = 'high'
        else:
            raise ValueError(f'Invalid risk assessment result: {prediction}')
        return fall_risk


class PDF(FPDF):
    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.WIDTH = 210
        self.HEIGHT = 297

    def lines(self):
        self.set_line_width(0.0)
        # bottom bar graph
        self.line(10.0, 235.0, self.WIDTH - 10, 235.0)  # top one
        self.line(10.0, 235.0 + self.HEIGHT / 5, self.WIDTH - 10, 235.0 + self.HEIGHT / 5)  # bottom one
        self.line(10.0, 235.0, 10.0, 235.0 + self.HEIGHT / 5)  # left one
        self.line(self.WIDTH - 10, 235.0, self.WIDTH - 10, 235.0 + self.HEIGHT / 5)  # right one

        self.line(76, 279.0 - 2 * HEIGHT / 4, 44, 300.0 - 2 * HEIGHT / 4)

    def texts(self, skdh_data):
        # Sleep report section
        self.set_font('Arial', '', 22)
        self.text(85, 180, "Sleep Report")
        # sleepScore
        self.set_xy(47.0, 258.0 - HEIGHT / 4)
        self.set_font('Arial', '', 14)
        self.multi_cell(0, 10, "Sleep Scores")
        # restlessness
        self.set_xy(44.0, 265.0 - HEIGHT / 4)
        # self.set_text_color(76.0, 32.0, 250.0)
        self.set_font('Arial', '', 9)
        self.multi_cell(0, 10, "Restlessness      Sleep")
        # index
        self.set_xy(49.0, 270.0 - HEIGHT / 4)
        self.set_font('Arial', '', 9)
        self.multi_cell(0, 10, "Index           Duration")

        # restlessness index and duration field
        interval = 0
        self.set_xy(53.0 + interval, 279.0 - HEIGHT / 4)
        self.set_font('Arial', '', 16)
        restless_ix = self.compute_sleep_hazard_index(skdh_data)
        ###
        # Set color for fall-risk rectangles
        if float(restless_ix) > 1.15:
            self.set_fill_color(255, 0, 0)
        elif 1.0 <= float(restless_ix) <= 1.15:
            self.set_fill_color(255, 255, 0)
        else:
            self.set_fill_color(0, 255, 0)
        high_style = 'DF' if float(restless_ix) > 1.15 else 'D'
        medium_style = 'DF' if 1.0 <= float(restless_ix) <= 1.15 else 'D'
        low_style = 'DF' if float(restless_ix) < 1.0 else 'D'
        self.rect(44, 278.0 - HEIGHT / 4, w=20, h=6, style=high_style)
        self.set_font('Arial', '', 12)
        self.text(49, 283.0 - HEIGHT / 4, "HIGH")
        self.rect(44, 285.0 - HEIGHT / 4, w=20, h=6, style=medium_style)
        self.text(46, 290.0 - HEIGHT / 4, "MEDIUM")
        self.rect(44, 292.0 - HEIGHT / 4, w=20, h=6, style=low_style)
        self.text(50, 297.0 - HEIGHT / 4, "LOW")
        self.set_fill_color(211, 211, 211)
        ###


        # self.multi_cell(0, 10, ri)
        interval += 18
        self.set_xy(50.0 + interval, 279.0 - HEIGHT / 4)
        self.set_font('Arial', '', 14)
        sd = self.compute_sleep_duration(skdh_data)
        self.multi_cell(0, 10, sd)
        self.text(50.0 + interval, 290.0 - HEIGHT / 4, "hours")

        # pie chart title
        self.set_xy(117.0, 255.0 - HEIGHT / 4)
        self.set_font('Arial', '', 18)
        self.multi_cell(0, 10, "Sleep Breakdown")

    def text_activity(self, name, skdh_data):
        # Activity Report Section
        self.set_font('Arial', '', 20)
        self.text(83, 115, "Activity Report")
        self.text(86, 241, "Daily Activity")
        # Minutes section
        self.set_xy(45.0, 268.0 - 2 * HEIGHT / 4)
        self.set_font('Arial', '', 14)
        self.multi_cell(0, 10, "Active Minutes")
        d = []
        for f_name in name:
            with open(f_name) as f:
                data = json.load(f)
                for k in data:
                    d.append(k)
        data = [k for k in d]
        self.set_xy(50.0, 279.0 - 2 * HEIGHT / 4)
        self.set_font('Arial', '', 18)
        # Active minutes section
        act_mins = self.compute_active_mins(skdh_data)
        self.multi_cell(0, 10, act_mins)
        self.set_xy(65.0, 286.0 - 2 * HEIGHT / 4)
        self.set_font('Arial', '', 16)
        # Recommended number of minutes
        self.multi_cell(0, 10, '30')
        self.set_font('Arial', '', 10)
        self.text(58, 297.0 - 2 * HEIGHT / 4, "recommended")
        self.text(63, 302.0 - 2 * HEIGHT / 4, "minutes")
        self.set_font('Arial', '', 18)
        self.text(115.0, 270.0 - 2 * HEIGHT / 4, "Activity Breakdown")

    def compute_active_mins(self, skdh_results):
        total = sum(skdh_results['act_metrics']['wake mod 5s epoch [min]']) + sum(skdh_results['act_metrics']['wake vig 5s epoch [min]'])
        return str(round(total / len(skdh_results['act_metrics']['wake mod 5s epoch [min]']), 2))

    def compute_sleep_duration(self, skdh_results):
        total = sum(skdh_results['sleep_metrics']['average sleep duration'])
        return str(round(total / len(skdh_results['sleep_metrics']['average sleep duration']) / 60.0, 2))

    def compute_sleep_hazard_index(self, skdh_results):
        total = sum(skdh_results['sleep_metrics']['sleep average hazard'])
        return str(round(total / len(skdh_results['sleep_metrics']['sleep average hazard']), 2))

    def page_body(self, images):
        # Determine how many plots there are per page and set positions
        # and margins accordingly
        if len(images) == 3:
            self.image(images[2], 125, 273 - 2 * HEIGHT / 4, 37, 37)
            self.image(images[1], 125, 265.0 - HEIGHT / 4, 37, 37)
            self.image(images[0], 0, 245, self.WIDTH, self.HEIGHT / 5 - 10)
            # self.image(images[0], -10, 235, self.WIDTH+20,self.HEIGHT/5)

    def print_page(self, images, skdh_data):
        # Generates the report
        self.page_body(images)
        self.texts(skdh_data)
        self.text_activity(['./digit.json', './digit.json'], skdh_data)
        self.lines()


def main():
    skdh_path = '/home/grainger/Desktop/skdh_testing/ml_model/input_metrics/skdh/skdh_results_20220815-171703.json'
    assessment_path = assessment_path = '/home/grainger/Desktop/test_risk_assessments/customers/customer_Grainger/site_Breed_Road/batch_0000000000000001_2022_08_25/assessment_0000000000000001_2022_08_25/'
    path_handler = PathHandler(assessment_path)
    path_handler.ra_metrics_file = '/home/grainger/Desktop/test_risk_assessments/customers/customer_Grainger/site_Breed_Road/batch_0000000000000001_2022_08_25/assessment_0000000000000001_2022_08_25/generated_data/ra_model_metrics/model_input_metrics_20220831-171046.json'
    path_handler.skdh_pipeline_results_file = '/home/grainger/Desktop/test_risk_assessments/customers/customer_Grainger/site_Breed_Road/batch_0000000000000001_2022_08_25/assessment_0000000000000001_2022_08_25/generated_data/skdh_pipeline_results/skdh_results_20220831-171046.json'
    path_handler.ra_results_file = ''
    ra_results = {
        'model_path': '',
        'scaler_path': '',
        'prediction': 0,
        'low-risk': 0,
        'moderate-risk': 1,
        'high-risk': 2
    }
    rg = ReportGenerator()
    rg.generate_report(path_handler, ra_results)


if __name__ == '__main__':
    main()
