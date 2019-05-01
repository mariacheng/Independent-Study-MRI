% Zone maps
% figure;
% for i = 1:size(CDI,3)
%     subplot(3,size(CDI,3)/3,i);
%     imagesc(casesTableArr.ZoneMap(:,:,i));
% end

% Slice maps with zones
zmap = casesTableArr.ZoneMap(60:80,60:90,:);
for z = 1:size(zmap,3)
    figure; imagesc(zmap(:,:,z));
end

x = imresize(T2(:,:,14),[144,144]);
y = x(60:80,60:90).*(casesTableArr.ZoneMap(60:80,60:90,14) > 0);

figure; subplot(2,1,1); imagesc(casesTableArr.ZoneMap(60:80,60:90,14));
subplot(2,1,2); imagesc(y);
subplot(2,1,2); imagesc(x(60:80,60:90));
colormap(gray)