import kivy
import os
from kivy.app import App
from kivy.uix.widget import Widget
from kivymd.app import MDApp
from kivy.core.window import Window
from kivymd.uix.dialog import MDDialog
from kivymd.uix.spinner import MDSpinner
from kivy.uix.actionbar import ActionButton
import ctypes

from kivy.properties import (
    StringProperty,
    ObjectProperty,
    BooleanProperty,
)
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder

# function
from train_model import train
from use_model import use_model
from search_text import search_text

# from search_logo import search_logo

# train an use model
import shutil
from plyer import filechooser


class HoverBehavior(object):
    hovered = BooleanProperty(False)
    border_point = ObjectProperty(None)

    def __init__(self, **kwargs):
        self.register_event_type("on_enter")
        self.register_event_type("on_leave")
        Window.bind(mouse_pos=self.on_mouse_pos)
        super(HoverBehavior, self).__init__(**kwargs)

    def on_mouse_pos(self, *args):
        if not self.get_root_window():
            return
        pos = args[1]
        inside = self.collide_point(*self.to_widget(*pos))
        if self.hovered == inside:
            return
        self.border_point = pos
        self.hovered = inside
        if inside:
            self.dispatch("on_enter")
        else:
            self.dispatch("on_leave")

    def on_enter(self):
        pass

    def on_leave(self):
        pass


from kivy.factory import Factory

Factory.register("HoverBehavior", HoverBehavior)


class MyActionButton(HoverBehavior, ActionButton):
    pass


class MoveWindow(BoxLayout):
    def on_touch_down(self, touch):
        print("\nISD.on_touch_down:")

        if self.collide_point(*touch.pos):
            print("\ttouch.pos =", touch.pos)
            self.touch_x, self.touch_y = touch.pos[0], touch.pos[1]
            print("BEFORE ==>", Window.top, Window.left)

            return True
        return super(MoveWindow, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        print("\nISD.on_touch_move:")

        if self.collide_point(*touch.pos):
            print("\ttouch.pos =", touch.pos)
            before_top = self.touch_y - Window.top
            before_left = self.touch_x - Window.left
            top = abs(touch.pos[1] - before_top)
            left = abs(touch.pos[0] - before_left)
            Window.top = top
            Window.left = left
            print("TOP ==>", Window.top)
            print("LEFT ==>", Window.left)

            return True
        return super(MoveWindow, self).on_touch_move(touch)


class Main(MDApp):
    id = ""
    loading = None
    found = None
    empty_file = None
    alert_path = None
    select_model = None
    export = None
    train = None
    training = None
    screeing = None
    count_path = 0
    name_model = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def file_chooser(self, id):
        selected = filechooser.choose_dir()
        self.name_model = ISD().name_model

        if selected != []:
            ISD().set_path(selected, id)
            if id == "path_images":
                self.root.ids.path_images.text = selected[0]

            else:
                if id == "path_train":
                    self.root.ids.path_train.text = selected[0]
                    self.root.ids.show_train.text = (
                        f"1  ภาพที่ใช้ในการฝึก  :  {selected[0]}"
                    )
                    self.root.ids.sum_train.text = (
                        f"1  ภาพที่ใช้ในการฝึก  :  {selected[0]}"
                    )
                    # self.count_path += 1

                if id == "path_test":
                    self.root.ids.path_test.text = selected[0]
                    self.root.ids.show_test.text = (
                        f"2  ภาพที่ใช้ในการทดสอบ  :  {selected[0]}"
                    )
                    self.root.ids.sum_test.text = (
                        f"2  ภาพที่ใช้ในการทดสอบ  :  {selected[0]}"
                    )
                    # self.count_path += 1

                if (
                    (ISD.name_model != "")
                    and (ISD.set_test != "")
                    and (ISD.set_train != "")
                ):
                    self.root.ids.next_step.disabled = False

    def build(self):
        Window.size = (1500, 900)
        Window.top = 50
        Window.left = 50
        self.theme_cls.theme_style = "Dark"
        Window.borderless = True
        self.icon = "/Image/title_bar/logo.png"
        # Window.set_icon("../Image/title_bar/logo.png")
        return ISD()

    def Minus_app_button(self):
        App.get_running_app().root_window.minimize()

    def close_app_button(self):
        app.stop()

    def collide_point(pos):
        inside = pos
        return inside

    def get_root_window(self):
        pass

    def alert_select_model(self):
        if not self.select_model:
            self.select_model = MDDialog(
                title="Please select a model!",
            )
        self.select_model.open()

    def alert_select_path(self):
        if not self.alert_path:
            self.alert_path = MDDialog(
                title="Please select the correct model and path!",
            )
        self.alert_path.open()

    def alert_train(self):
        if not self.train:
            self.train = MDDialog(
                title="Please check model name or select path.!",
            )
        self.train.open()

    def alert_succeed(self):
        if not self.screeing:
            self.screeing = MDDialog(
                title="Completed.",
            )
        self.screeing.open()

    def alert_empty_file(self):
        if not self.empty_file:
            self.empty_file = MDDialog(title="This folder does not contain images.")
        self.empty_file.open()

    def alert_found(self):
        if not self.found:
            self.found = MDDialog(title="Please check the image before exporting it.")
        self.found.open()

    def alert_export(self):
        if not self.export:
            self.export = MDDialog(title="Please export again.")
        self.export.open()

    def alert_loading(self):
        if not self.loading:
            self.loading = MDDialog(
                # title="Inspecting...",
                type="custom",
                width=50,
                height=50,
                content_cls=MDSpinner(
                    size_hint=(None, None),
                    # size=(70, 70),
                    width=100,
                    height=100,
                    pos_hint={"center_x": 0.5, "center_y": 0.5},
                ),
            )
        self.loading.open()


class ISD(Widget):

    font = "fonts/Kanit.ttf"
    templates_path = "INS/DATASET/LOGO/"
    path_models = StringProperty(os.listdir("INS/models/"))
    result_model = None
    path = ""
    accuracy = 0
    loss = 0
    accuracy_txt = f"{accuracy}%"
    name_model = StringProperty("")
    current_user = os.getlogin()
    path_export = "C:/Users/{}/Pictures/".format(current_user)
    set_train = StringProperty("")
    set_test = StringProperty("")
    set_images = StringProperty("")
    t_train = StringProperty("")
    t_test = StringProperty("")
    next_step = StringProperty("step2")
    list_txt = []
    predict_images = {}
    final_predict = {}

    def select_model(self, value):
        self.ids.model.text = value

    # TODO: Function Train Model
    def get_data_for_train(self, name):

        if name and set_test and set_train:
            global name_model
            name_model = name
            self.ids.namemodel.text = f"ชื่อโมเดล  {name}"
            self.ids.name.text = f"ชื่อโมเดล  {name}"
            list_train = [
                set_train + f
                for f in os.listdir(set_train)
                if os.listdir(os.path.join(set_train, f))
            ]
            list_test = [
                set_test + f
                for f in os.listdir(set_test)
                if os.listdir(os.path.join(set_test, f))
            ]

            if (len(list_test) == 2) and (len(list_train) == 2):
                self.ids.show_sub_train1.text = list_train[0]
                self.ids.show_sub_train2.text = list_train[1]
                self.ids.show_sub_test1.text = list_test[0]
                self.ids.show_sub_test2.text = list_test[1]
                self.ids.step_trainng.current = "step2"

            else:
                app.alert_train()

        else:
            app.alert_train()
            self.ids.step_trainng.current = "step1"

    def train_model(self):
        # print(name_model)
        global accuracy
        global loss
        global precision
        global recall
        global accuracy_txt
        global result_model
        global path_save
        global t_train
        global t_test
        try:
            (
                model,
                get_accuracy,
                get_loss,
                get_precision,
                get_recall,
                total_train,
                total_test,
            ) = train(set_train, set_test)
            # print("model :", model)
            # print("accuracy :", get_accuracy)
            # print("loss :", get_loss)
            # print("total train :", total_train)
            # print("total test :", total_test)
            result_model = model
            path_save = f"INS/models/{name_model}.h5"
            # model.save(path_save)
            print()
            loss = "{:.5f}".format(get_loss)
            # loss = get_loss
            accuracy = "{:.4f}".format(get_accuracy)
            # loss = int(float(loss) * 100)
            precision = "{:.4f}".format(get_precision)
            recall = "{:.4f}".format(get_recall)
            accuracy = float(float(accuracy) * 100)
            accuracy_txt = f"{accuracy}%"
            # loss_txt = f"{loss}"
            t_train = total_train
            t_test = total_test
            print(accuracy_txt)
            print(loss)
            self.ids.loss_txt.text = f"Loss  :  {loss}"
            self.ids.accuracy_txt.text = f"Accuracy  :  {accuracy_txt}"
            self.ids.pre_txt.text = f"Precision  :  {precision}"
            self.ids.rec_txt.text = f"Recall  :  {recall}"
            self.ids.total_train.text = f"ภาพฝึกทั้งหมด  :  {t_train}"
            self.ids.total_test.text = f"ภาพทดสอบทั้งหมด  :  {t_test}"
            self.ids.fig_loss.source = "Image/plot_fig/loss.png"
            self.ids.fig_acc.source = "Image/plot_fig/accuracy.png"
            self.ids.fig_rec.source = "Image/plot_fig/recall.png"
            self.ids.fig_pre.source = "Image/plot_fig/precision.png"
            self.ids.accuracy.value = accuracy
            self.ids.step_trainng.current = "step3"
            app.alert_succeed()

        except Exception as e:
            print(e)
            app.alert_select_path()
            self.ids.step_trainng.current = "step1"

    # TODO:SET PATH FOR USE ALL FUNCTION
    def set_path(self, selected, id):
        # global path
        # path = p
        global set_test
        global set_images
        global set_train
        # print("===", selected, "===")
        # print("id : ", id)
        if id == "path_images":
            # self.set_images = selected[0]
            set_images = selected[0]
            print("set_images : ", set_images)

        else:
            if id == "path_train":
                # self.set_train = selected[0]
                set_train = selected[0] + "\\"
                # print("set_train : ", set_train)

            elif id == "path_test":
                # self.path_test = selected[0]
                set_test = selected[0] + "\\"
                # print("set_test : ", set_test)

    # MERGE DICT
    def merge_dict(self, dict_1, dict_2):
        result = {}
        for predict1 in dict_1.keys():
            if "true" in dict_1[predict1]:
                for predict2 in dict_2.keys():
                    if predict2 == predict1:
                        if "true" in dict_2[predict2]:
                            result[predict2] = "true"

                        elif "false" in dict_2[predict2]:
                            result[predict1] = "false"

            elif "false" in dict_1[predict1]:
                result[predict1] = "false"
        return result

    # TODO: Function Saerch Text
    def saerch_txt(self, path, text):
        text_images, img_text_req = search_text(path, text)
        print("text_images", text_images)
        print("img_text_req", img_text_req)
        text_images = self.merge_dict(text_images, img_text_req)
        return text_images

    # TODO: Function Screening Model
    def screening_model(self, model):

        global final_predict
        global predict_images
        global set_images
        number_of_path = []
        try:
            if model != "เลือกโมเดล" and set_images:
                for item in os.listdir(set_images):
                    if item.lower().endswith((".png", ".jpg", ".gif", ".bmp")):
                        number_of_path.append(item)

                if number_of_path:
                    base_model = "INS/models/" + model
                    predict_images = use_model(set_images, base_model)
                    text_images = self.saerch_txt(set_images, self.ids.txt_saerch.text)
                    final_predict = self.merge_dict(predict_images, text_images)
                    # final_predict = predict_images
                    print("final_predict ===>", final_predict)
                    self.reset_date()
                    # self.ids.model.text = "เลือกโมเดล"
                    # self.ids.txt_saerch.text = ""
                    # self.ids.path_images.text = ""
                    # set_images = ""
                    app.alert_succeed()

                else:
                    app.alert_empty_file()

            else:
                print("กรุณาเลือกโมเดลและโฟลเดอร์")
                app.alert_select_path()

        except Exception as e:
            print(e)
            app.alert_select_path()

    # TODO : Export folder
    def get_value_export(self, name):

        try:
            if name and final_predict:
                self.export_folder(name, final_predict)

        except Exception as err:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(err).__name__, err.args)
            print(message)
            app.alert_found()

    def export_folder(self, name, images_predict):

        try:
            if images_predict:

                if name == "not_slip":
                    _export = os.path.join(self.path_export, "Not Slip/")
                    os.mkdir(_export)

                    for predict in images_predict.keys():
                        if "false" in images_predict[predict]:
                            shutil.copy(predict, self.path_export + "Not Slip/")

                    open_folder = os.path.realpath(self.path_export + "Not Slip/")
                    os.startfile(open_folder)

                elif name == "slip":
                    _export = os.path.join(self.path_export, "Slip/")
                    os.mkdir(_export)
                    for predict in images_predict.keys():
                        if "true" in images_predict[predict]:
                            shutil.copy(predict, self.path_export + "Slip/")

                    open_folder = os.path.realpath(self.path_export + "Slip/")
                    os.startfile(open_folder)

            else:
                app.alert_found()

        except Exception as err:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(err).__name__, err.args)
            print(message)

            if err.args != (17, "Cannot create a file when that file already exists"):
                app.alert_found()

            else:
                # app.alert_export()
                self.export_images(name, images_predict)

    def export_images(self, name, images_predict):

        print(name)
        if name == "not_slip":
            _export = os.path.join(self.path_export, "Not Slip/")
            shutil.rmtree(_export)
            os.mkdir(_export)

            for predict in images_predict.keys():
                if "false" in images_predict[predict]:
                    shutil.copy(predict, self.path_export + "Not Slip/")

            open_folder = os.path.realpath(self.path_export + "Not Slip/")
            os.startfile(open_folder)

        elif name == "slip":
            _export = os.path.join(self.path_export, "Slip/")
            shutil.rmtree(_export)
            os.mkdir(_export)

            for predict in images_predict.keys():
                if "true" in images_predict[predict]:
                    shutil.copy(predict, self.path_export + "Slip/")

            open_folder = os.path.realpath(self.path_export + "Slip/")
            os.startfile(open_folder)

    # TODO: Function Remove Model
    def remove_model(self, model):
        if model != "เลือกโมเดล":
            global path_models
            os.remove("INS/models/" + model)
            path_models = os.listdir("INS/models/")
            self.ids.model.values = path_models
            self.ids.model.text = "เลือกโมเดล"
            self.reset_date()
        else:
            print("กรุณาเลือกโมเดล")
            app.alert_select_model()

    # TODO:Export Model
    def export_model(self):
        result_model.save(path_save)
        self.reset_date()

    def reset_date(self):
        global set_test
        global set_train
        global set_images
        global path_models
        path_models = os.listdir("INS/models/")
        self.ids.model.values = path_models
        app.count_path = 0
        self.ids.next_step.disabled = True
        self.ids.name_model.text = ""
        self.ids.namemodel.text = ""
        self.ids.name.text = ""
        self.ids.path_train.text = ""
        self.ids.show_train.text = ""
        self.ids.sum_train.text = ""
        self.ids.path_test.text = ""
        self.ids.show_test.text = ""
        self.ids.sum_test.text = ""
        set_test = ""
        set_train = ""
        self.ids.model.text = "เลือกโมเดล"
        self.ids.txt_saerch.text = ""
        self.ids.path_images.text = ""
        set_images = ""
        # os.remove("Image/plot_fig/fig1.png")


if __name__ == "__main__":
    app = Main()
    app.run()
