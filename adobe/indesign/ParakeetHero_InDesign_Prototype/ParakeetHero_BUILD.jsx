#target indesign

(function () {
    var previousInteraction = app.scriptPreferences.userInteractionLevel;
    app.scriptPreferences.userInteractionLevel = UserInteractionLevels.NEVER_INTERACT;

    var root = File($.fileName).parent;
    var outputFolder = new Folder(root.fsName + "/output_final");
    if (!outputFolder.exists) { outputFolder.create(); }
    var logFile = new File(outputFolder.fsName + "/build.log");
    var logLines = [];

    function log(message) { logLines.push(new Date().toString() + " " + message); }
    function writeLog() {
        logFile.encoding = "UTF-8";
        logFile.open("w");
        logFile.write(logLines.join("\n") + "\n");
        logFile.close();
    }
    function safeName(item, label) {
        if (!label) { return; }
        try { item.name = label; } catch (e) {}
        try { item.label = label; } catch (e2) {}
    }
    function clearFrameAppearance(item) {
        // InDesign 2026 can coerce a generic Swatch object into an anonymous
        // black Color when assigned through ExtendScript. The DOM also accepts
        // a String, which preserves the special None swatch correctly.
        item.strokeColor = "None";
        item.strokeWeight = 0;
    }
    function ensureLayer(doc, name) {
        var layer = doc.layers.itemByName(name);
        if (!layer.isValid) { layer = doc.layers.add({name: name}); }
        return layer;
    }
    function ensureRgb(doc, name, values) {
        var color = doc.colors.itemByName(name);
        if (!color.isValid) {
            color = doc.colors.add({
                name: name,
                model: ColorModel.PROCESS,
                space: ColorSpace.RGB,
                colorValue: values
            });
        }
        return color;
    }
    function addRect(page, layer, bounds, fillSwatch, label, opacity) {
        var r = page.rectangles.add();
        r.itemLayer = layer;
        r.geometricBounds = bounds;
        r.strokeColor = noneSwatch;
        r.strokeWeight = 0;
        r.fillColor = fillSwatch || noneSwatch;
        if (opacity !== undefined) {
            r.transparencySettings.blendingSettings.opacity = opacity;
        }
        clearFrameAppearance(r);
        safeName(r, label);
        return r;
    }
    function addGraphicFrame(page, layer, bounds, fileObj, label, fitMode) {
        if (!fileObj.exists) { throw new Error("Missing linked asset: " + fileObj.fsName); }
        var r = page.rectangles.add();
        r.itemLayer = layer;
        r.geometricBounds = bounds;
        r.strokeColor = noneSwatch;
        r.strokeWeight = 0;
        r.fillColor = noneSwatch;
        safeName(r, label);
        r.place(fileObj);
        r.fit(fitMode || FitOptions.FILL_PROPORTIONALLY);
        r.fit(FitOptions.CENTER_CONTENT);
        r.fillColor = "None";
        clearFrameAppearance(r);
        return r;
    }
    function styleText(textObj, family, style, fallback, pointSize, fill, tracking) {
        try {
            textObj.appliedFont = family;
            textObj.fontStyle = style;
        } catch (e) {
            textObj.appliedFont = fallback;
            textObj.fontStyle = "Bold";
        }
        textObj.pointSize = pointSize;
        textObj.leading = pointSize * 0.92;
        textObj.fillColor = fill;
        textObj.tracking = tracking || 0;
    }
    function addText(page, layer, bounds, contents, label, pointSize, fill, style, justification, tracking) {
        var t = page.textFrames.add();
        t.itemLayer = layer;
        t.geometricBounds = bounds;
        t.strokeColor = noneSwatch;
        t.strokeWeight = 0;
        t.fillColor = "None";
        t.contents = contents;
        t.textFramePreferences.insetSpacing = [0, 0, 0, 0];
        t.textFramePreferences.verticalJustification = VerticalJustification.CENTER_ALIGN;
        t.parentStory.paragraphs.everyItem().justification = justification || Justification.LEFT_ALIGN;
        t.parentStory.paragraphs.everyItem().hyphenation = false;
        styleText(t.parentStory.texts[0], "Montserrat", style || "Black", "Arial", pointSize, fill, tracking);
        t.fillColor = noneSwatch;
        clearFrameAppearance(t);
        safeName(t, label);
        return t;
    }
    function addRule(page, layer, bounds, strokeSwatch, weight, label) {
        var line = page.graphicLines.add();
        line.itemLayer = layer;
        line.geometricBounds = bounds;
        line.strokeColor = strokeSwatch;
        line.strokeWeight = weight;
        safeName(line, label);
        return line;
    }
    function assertNoOverset(doc) {
        var overset = [];
        for (var i = 0; i < doc.textFrames.length; i++) {
            if (doc.textFrames[i].overflows) { overset.push(doc.textFrames[i].label || ("textFrame[" + i + "]")); }
        }
        if (overset.length) { throw new Error("Overset text: " + overset.join(", ")); }
    }
    function configurePng(pageName) {
        app.pngExportPreferences.pngColorSpace = PNGColorSpaceEnum.RGB;
        app.pngExportPreferences.pngQuality = PNGQualityEnum.MAXIMUM;
        app.pngExportPreferences.exportResolution = 72;
        app.pngExportPreferences.transparentBackground = false;
        app.pngExportPreferences.antiAlias = true;
        app.pngExportPreferences.exportingSpread = false;
        app.pngExportPreferences.pngExportRange = PNGExportRangeEnum.EXPORT_RANGE;
        app.pngExportPreferences.pageString = pageName;
    }
    function buildVariant(cfg) {
        log("BUILD START " + cfg.id);
        var doc = app.documents.add();
        doc.documentPreferences.facingPages = false;
        doc.viewPreferences.horizontalMeasurementUnits = MeasurementUnits.POINTS;
        doc.viewPreferences.verticalMeasurementUnits = MeasurementUnits.POINTS;
        doc.documentPreferences.pageWidth = cfg.width + "pt";
        doc.documentPreferences.pageHeight = cfg.height + "pt";
        doc.documentPreferences.documentBleedTopOffset = 0;
        doc.documentPreferences.documentBleedBottomOffset = 0;
        doc.documentPreferences.documentBleedInsideOrLeftOffset = 0;
        doc.documentPreferences.documentBleedOutsideOrRightOffset = 0;

        var page = doc.pages[0];
        noneSwatch = doc.swatches.item("None");
        if (!noneSwatch.isValid) { noneSwatch = doc.swatches.itemByName("None"); }
        if (!noneSwatch.isValid) { throw new Error("Required InDesign None swatch not found"); }
        log("NONE SWATCH " + noneSwatch.name + " " + noneSwatch.toSpecifier());
        var ink = ensureRgb(doc, "Parakeet Ink", [4, 11, 18]);
        var white = ensureRgb(doc, "Parakeet White", [247, 250, 252]);
        var gold = ensureRgb(doc, "Parakeet Gold", [255, 193, 59]);
        var cyan = ensureRgb(doc, "Parakeet Cyan", [38, 218, 229]);
        var muted = ensureRgb(doc, "Parakeet Muted", [173, 190, 199]);

        var backgroundLayer = ensureLayer(doc, "00_BACKGROUND");
        var atmosphereLayer = ensureLayer(doc, "10_ATMOSPHERE");
        var heroLayer = ensureLayer(doc, "20_HERO");
        var surfaceLayer = ensureLayer(doc, "30_TEXT_SURFACE");
        var headlineLayer = ensureLayer(doc, "40_HEADLINE");
        var metadataLayer = ensureLayer(doc, "50_BRAND_METADATA");

        addGraphicFrame(page, backgroundLayer, [0, 0, cfg.height, cfg.width], new File(root.fsName + "/" + cfg.background), "background.vector", FitOptions.FILL_PROPORTIONALLY);
        addGraphicFrame(page, atmosphereLayer, [0, 0, cfg.height, cfg.width], new File(root.fsName + "/" + cfg.atmosphere), "atmosphere.vector", FitOptions.FILL_PROPORTIONALLY);
        addGraphicFrame(page, heroLayer, cfg.heroBounds, new File(root.fsName + "/assets/images/leo_bigger_icf_ai_astonished_revelation_v1.png"), "hero.leo_bigger", cfg.heroFit);

        addRect(page, surfaceLayer, cfg.surfaceBounds, ink, "surface.headline", 76);
        addRect(page, surfaceLayer, cfg.accentBounds, gold, "accent.gold", 100);
        addRule(page, surfaceLayer, cfg.cyanRule, cyan, cfg.ruleWeight, "accent.cyan_rule");

        addText(page, headlineLayer, cfg.headline1, "WUNDER", "headline.wunder", cfg.size1, gold, "Black", Justification.LEFT_ALIGN, -16);
        addText(page, headlineLayer, cfg.headline2, "SIND NICHT", "headline.sind_nicht", cfg.size2, white, "Black", Justification.LEFT_ALIGN, 4);
        addText(page, headlineLayer, cfg.headline3, "VORBEI", "headline.vorbei", cfg.size3, cyan, "Black", Justification.LEFT_ALIGN, -12);

        addRect(page, metadataLayer, cfg.brandPill, ink, "brand.pill", 86);
        addText(page, metadataLayer, cfg.brandText, "ICF Z\u00dcRICH", "brand.icf_zuerich", cfg.brandSize, white, "SemiBold", Justification.CENTER_ALIGN, 90);
        addText(page, metadataLayer, cfg.speakerText, "LEO BIGGER", "speaker.leo_bigger", cfg.metaSize, muted, "SemiBold", Justification.LEFT_ALIGN, 115);
        addText(page, metadataLayer, cfg.badgeText, "PARAKEET HERO", "badge.parakeet_hero", cfg.badgeSize, gold, "SemiBold", Justification.RIGHT_ALIGN, 115);

        assertNoOverset(doc);

        var indd = new File(outputFolder.fsName + "/" + cfg.output + ".indd");
        var idml = new File(outputFolder.fsName + "/" + cfg.output + ".idml");
        var png = new File(outputFolder.fsName + "/" + cfg.output + ".png");
        doc.save(indd);
        doc.exportFile(ExportFormat.INDESIGN_MARKUP, idml, false);
        configurePng(page.name);
        doc.exportFile(ExportFormat.PNG_FORMAT, png, false);
        doc.close(SaveOptions.NO);
        log("BUILD OK " + cfg.id + " -> " + png.fsName);
    }

    var noneSwatch;
    try {
        buildVariant({
            id: "vertical_9x16",
            width: 1080,
            height: 1920,
            output: "ParakeetHero_9x16",
            background: "assets/vectors/background_9x16.svg",
            atmosphere: "assets/vectors/atmosphere_9x16.svg",
            heroBounds: [0, 0, 1920, 1080],
            heroFit: FitOptions.FILL_PROPORTIONALLY,
            surfaceBounds: [344, 466, 1092, 1060],
            accentBounds: [400, 500, 418, 644],
            cyanRule: [1032, 502, 1032, 1008],
            ruleWeight: 4,
            headline1: [438, 500, 606, 1040],
            headline2: [596, 500, 754, 1040],
            headline3: [740, 500, 928, 1040],
            size1: 104,
            size2: 72,
            size3: 110,
            brandPill: [84, 704, 178, 1038],
            brandText: [94, 722, 168, 1020],
            brandSize: 32,
            speakerText: [1112, 504, 1180, 826],
            metaSize: 27,
            badgeText: [1112, 820, 1180, 1038],
            badgeSize: 22
        });
        buildVariant({
            id: "youtube_16x9",
            width: 1280,
            height: 720,
            output: "ParakeetHero_16x9",
            background: "assets/vectors/background_16x9.svg",
            atmosphere: "assets/vectors/atmosphere_16x9.svg",
            heroBounds: [0, 0, 720, 610],
            heroFit: FitOptions.PROPORTIONALLY,
            surfaceBounds: [108, 544, 626, 1252],
            accentBounds: [148, 588, 160, 744],
            cyanRule: [574, 590, 574, 1184],
            ruleWeight: 3,
            headline1: [172, 588, 276, 1218],
            headline2: [270, 588, 368, 1218],
            headline3: [354, 588, 472, 1218],
            size1: 76,
            size2: 52,
            size3: 86,
            brandPill: [38, 974, 102, 1246],
            brandText: [46, 992, 94, 1228],
            brandSize: 24,
            speakerText: [492, 590, 544, 870],
            metaSize: 22,
            badgeText: [492, 910, 544, 1218],
            badgeSize: 18
        });
        log("ALL BUILDS OK");
        writeLog();
    } catch (error) {
        log("BUILD FAILED " + error.message + " line=" + error.line);
        writeLog();
        app.scriptPreferences.userInteractionLevel = previousInteraction;
        throw error;
    }
    app.scriptPreferences.userInteractionLevel = previousInteraction;
})();
