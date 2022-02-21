from src.motion_analysis.gait_analysis.gait_analyzer import GaitAnalyzer


class TestGaitAnalyzerV1:

    def test_estimate_vertical_displacement(self):
        # Need vertical acceleration signal: a list of acceleration values
        v_acc = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        # Sampling frequency of the vertical acceleration
        samp_freq = 1.0
        # Indexes of the start and stop of steps
        step_start_ix = 2
        step_end_ix = 7
        ga = GaitAnalyzer()
        pos = ga.estimate_v_displacement(v_acc, step_start_ix,
                                      step_end_ix, samp_freq, hpf=False)
        assert pos == [0.0, 0.0, 1.0, 3.0, 6.0]


def main():
    tester = TestGaitAnalyzerV1()
    tester.test_estimate_vertical_displacement()

if __name__ == '__main__':
    main()
